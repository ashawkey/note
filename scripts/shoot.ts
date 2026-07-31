import { chromium } from 'playwright'
import { mkdir, rm } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { join } from 'node:path'

import { serveDir } from './serve.ts'

/*
 * Screenshots, for the half of correctness a build cannot reach.
 *
 * A successful build tells you nothing about how it looks: a pane can be a hairline out of
 * alignment, the sheet's gloss can fall off in the wrong place, or the drawer can cover the
 * note it was meant to sit beside. Reviewing these is how you find that.
 *
 * Both palettes, every page, three widths — the breakpoints are where the layout actually
 * changes: contents pane dropped at 78rem, panel collapsed to a drawer at 56rem.
 */

const root = fileURLToPath(new URL('..', import.meta.url))
const outDir = join(root, 'test/shots')

const VIEWPORTS = [
  { name: 'wide', width: 1440, height: 1000 },
  { name: 'medium', width: 1100, height: 900 },
  { name: 'narrow', width: 560, height: 900 },
]

const PAGES = [
  { name: 'home', path: '' },
  { name: 'section', path: 'algorithm/' },
  { name: 'note', path: 'algorithm/basic/DA05_BinaryTree/' },
]

async function main(): Promise<void> {
  await rm(outDir, { recursive: true, force: true })
  await mkdir(outDir, { recursive: true })

  const server = await serveDir(join(root, 'dist'))
  const browser = await chromium.launch({ channel: 'chrome' })
  let written = 0

  try {
    for (const theme of ['light', 'dark'] as const) {
      for (const viewport of VIEWPORTS) {
        const context = await browser.newContext({
          viewport: { width: viewport.width, height: viewport.height },
          deviceScaleFactor: 2,
          colorScheme: theme,
        })
        const page = await context.newPage()

        for (const target of PAGES) {
          await page.goto(`${server.url}${target.path}`, { waitUntil: 'networkidle' })
          // Webfonts settle after networkidle on a cold context; without this the first
          // shot of each run is measurably different from the rest.
          await page.evaluate(() => document.fonts.ready)

          await page.screenshot({
            path: join(outDir, `${theme}-${viewport.name}-${target.name}.png`),
            fullPage: true,
          })
          written += 1
        }

        // The drawer only exists at the narrow breakpoint, and only once opened.
        if (viewport.name === 'narrow') {
          await page.goto(`${server.url}algorithm/basic/DA05_BinaryTree/`, { waitUntil: 'networkidle' })
          await page.click('[data-drawer-toggle]')
          await page.waitForTimeout(400) // the 260ms slide, plus a margin
          await page.screenshot({ path: join(outDir, `${theme}-narrow-drawer.png`) })
          written += 1
        }

        // Search is a pane on the field like any other and needs looking at.
        if (viewport.name === 'wide') {
          await page.goto(`${server.url}algorithm/basic/DA05_BinaryTree/`, { waitUntil: 'networkidle' })
          await page.click('[data-search-open]')
          await page.fill('[data-search-input]', 'quicksort')
          await page.waitForSelector('.search-result', { timeout: 5000 })
          await page.screenshot({ path: join(outDir, `${theme}-wide-search.png`) })
          written += 1
        }

        await context.close()
      }
    }
  } finally {
    await browser.close()
    await server.close()
  }

  console.log(`${written} screenshots written to test/shots/`)
}

await main()
