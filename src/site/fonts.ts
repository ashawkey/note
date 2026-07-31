import { readFile } from 'node:fs/promises'

/*
 * Which webfont subsets to preload.
 *
 * LXGW WenKai ships as ~194 `unicode-range` subsets, which is what makes a CJK family
 * affordable at all: a page downloads the ranges it sets and no more. The cost is that the
 * browser cannot know which ones it needs until it has laid the text out, so the first paint
 * of a cold visit is always in the fallback face and swaps a moment later.
 *
 * A `<link rel="preload">` fixes that for the ranges every page is certain to use — the ASCII
 * that appears in prose, headings, code and the chrome alike — by starting those fetches with
 * the stylesheet rather than after layout. The CJK subsets are deliberately left alone: which
 * ones a note needs depends entirely on the note, and preloading them all would download
 * megabytes to save a flash.
 *
 * Derived from the stylesheet rather than hardcoded, because the subset numbering is an
 * artefact of how the family was built and would silently drift on the next font update.
 */

/** Codepoints that any page of prose is certain to contain. */
const ALWAYS_USED = [0x41, 0x61, 0x20, 0x2e] // 'A', 'a', space, full stop

interface FontFace {
  url: string
  weight: number
}

/**
 * Parse `@font-face` blocks and return the subsets covering ASCII, nearest-weight first.
 *
 * Only exact `U+xxxx` and `U+xxxx-yyyy` forms are understood. Wildcard ranges (`U+4??`) are
 * skipped rather than guessed at — the vendored stylesheet uses none, and a wrong guess would
 * preload a file no page needs.
 */
export async function latinSubsets(cssPath: string): Promise<string[]> {
  const css = await readFile(cssPath, 'utf8')
  const faces: FontFace[] = []

  for (const block of css.split('@font-face').slice(1)) {
    const src = /url\(['"]?([^'")]+)['"]?\)/.exec(block)?.[1]
    const ranges = /unicode-range:\s*([^;}]+)/.exec(block)?.[1]
    const weight = Number(/font-weight:\s*(\d+)/.exec(block)?.[1] ?? '400')
    if (!src || !ranges) continue

    const covers = ranges.split(',').some((part) => {
      const match = /^U\+([0-9a-f]+)(?:-([0-9a-f]+))?$/i.exec(part.trim())
      if (!match?.[1]) return false
      const low = Number.parseInt(match[1], 16)
      const high = match[2] ? Number.parseInt(match[2], 16) : low
      return ALWAYS_USED.some((code) => code >= low && code <= high)
    })

    if (covers) faces.push({ url: src.replace(/^\.\//, ''), weight })
  }

  // Regular before bold: body text is far more of the page than headings are.
  faces.sort((a, b) => Math.abs(a.weight - 400) - Math.abs(b.weight - 400))

  /*
   * Capped, because a preload is a promise that the file is needed immediately — and a
   * browser that is told everything is urgent schedules nothing well. Four covers regular and
   * bold across the two ASCII ranges the family splits them into.
   */
  return [...new Set(faces.map((face) => face.url))].slice(0, 4)
}
