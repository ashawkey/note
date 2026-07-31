import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { join } from 'node:path'

import { serveDir } from './serve.ts'

/*
 * Serve an existing dist/ without rebuilding it.
 *
 * `npm run dev` is what you usually want; this is for looking at a build you already have —
 * checking a deploy artifact, or serving while a slow rebuild runs elsewhere.
 */

const root = fileURLToPath(new URL('..', import.meta.url))
const dist = join(root, 'dist')

if (!existsSync(join(dist, 'index.html'))) {
  console.error('dist/ is empty - run `npm run build` first.')
  process.exit(1)
}

// A subpath by default, because that is the deploy shape most likely to expose an absolute
// URL that should have been relative.
const server = await serveDir(dist, '/note/', 4174)
console.log(`  Serving dist/ at ${server.url}`)
console.log('  Mounted on a subpath, so an absolute URL would break here. Ctrl-C to stop.')
