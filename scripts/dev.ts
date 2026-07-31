import { spawn } from 'node:child_process'
import { watch } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { join } from 'node:path'

import { serveDir } from './serve.ts'

/*
 * Local preview: build, serve, and rebuild when anything changes.
 *
 * Deliberately a full rebuild rather than incremental. The whole build is well under a
 * second for a few hundred notes, and every page embeds the navigation tree — so a new note
 * changes the sidebar of every other page anyway. An incremental path would have to know
 * that, and would be wrong the day it forgot.
 *
 * All three phases run, search included, so what you preview is what deploys.
 */

const root = fileURLToPath(new URL('..', import.meta.url))
const PORT = 4173

let running = false
let queued = false

function build(): Promise<void> {
  return new Promise((resolve) => {
    const child = spawn('npm', ['run', 'build'], {
      cwd: root,
      shell: true,
      stdio: ['ignore', 'pipe', 'inherit'],
    })
    // Only the interesting lines: the page count and any warnings. The three phase banners
    // on every keystroke would bury them.
    child.stdout.setEncoding('utf8')
    child.stdout.on('data', (chunk: string) => {
      for (const line of chunk.split('\n')) {
        if (/pages from|formula|no grammar|error/i.test(line)) process.stdout.write(`  ${line.trim()}\n`)
      }
    })
    child.on('close', () => resolve())
  })
}

async function rebuild(reason: string): Promise<void> {
  if (running) {
    queued = true
    return
  }
  running = true
  const started = Date.now()
  // ASCII only: this is the one script whose output a Windows console shows verbatim, and
  // cmd.exe's default code page turns an em dash or ellipsis into mojibake.
  process.stdout.write(`\n${reason} - rebuilding...\n`)
  await build()
  process.stdout.write(`  done in ${Date.now() - started}ms\n`)
  running = false

  if (queued) {
    queued = false
    await rebuild('more changes')
  }
}

await rebuild('initial build')

const server = await serveDir(join(root, 'dist'), '/', PORT)
console.log(`\n  Notes running at ${server.url}`)
console.log('  Watching notes/ and src/. Ctrl-C to stop.\n')

let timer: NodeJS.Timeout | undefined

function onChange(filename: string | null): void {
  // Editors write a file several times in a row; one rebuild per burst is enough.
  clearTimeout(timer)
  timer = setTimeout(() => void rebuild(filename ? `changed ${filename}` : 'changed'), 150)
}

for (const dir of ['notes', 'src']) {
  watch(join(root, dir), { recursive: true }, (_event, filename) => onChange(filename))
}
// A single file, so no `recursive` — passing it for a non-directory throws on Windows.
watch(join(root, 'site.config.ts'), (_event, filename) => onChange(filename))
