import { createReadStream } from 'node:fs'
import { stat } from 'node:fs/promises'
import { createServer, type Server } from 'node:http'
import { extname, join, normalize, sep } from 'node:path'

/*
 * A static server for the verification scripts.
 *
 * In-process rather than shelling out, so a check cannot leave a port bound after a failure,
 * and so the mount prefix can be varied — serving dist/ at `/note/` is how the checks prove
 * the relative-path scheme really does survive a project-page subpath deploy, which is the
 * one property that would otherwise only be discovered after publishing.
 */

const TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
  '.ttf': 'font/ttf',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.wasm': 'application/wasm',
}

export interface StaticServer {
  /** Base URL including the mount prefix, with a trailing slash. */
  url: string
  close(): Promise<void>
}

/**
 * Serve `dir` at `prefix`.
 *
 * @param prefix Mount path, e.g. `/` or `/note/`.
 * @param port   0 picks a free port, which is what the checks want so parallel runs cannot
 *               collide. `npm run dev` passes a fixed one so the URL stays bookmarkable.
 */
export async function serveDir(dir: string, prefix = '/', port = 0): Promise<StaticServer> {
  const mount = prefix.endsWith('/') ? prefix : `${prefix}/`

  const server: Server = createServer((req, res) => {
    const url = new URL(req.url ?? '/', 'http://localhost')
    let path = decodeURIComponent(url.pathname)

    if (!path.startsWith(mount)) {
      res.writeHead(404).end('outside mount')
      return
    }
    path = path.slice(mount.length - 1)

    // Contain traversal: resolve, then confirm the result is still inside dir.
    const resolved = join(dir, normalize(path))
    if (resolved !== dir && !resolved.startsWith(dir + sep)) {
      res.writeHead(403).end('forbidden')
      return
    }

    void (async () => {
      let file = resolved
      try {
        if ((await stat(file)).isDirectory()) file = join(file, 'index.html')
      } catch {
        res.writeHead(404).end('not found')
        return
      }

      try {
        await stat(file)
      } catch {
        res.writeHead(404).end('not found')
        return
      }

      /*
       * Cache the way a real host does, or local preview lies about how the site behaves.
       *
       * `no-store` on everything is the obvious choice for a dev server and it is actively
       * misleading here: it makes the browser re-fetch the webfont subsets on every single
       * navigation, so every page change shows a flash of the fallback face — a problem that
       * exists only because of the header. Fonts, hashed bundles and the search index never
       * change without their contents changing, so they are cached hard; HTML and the
       * generated stylesheet must revalidate, or a rebuild would not show up.
       */
      const extension = extname(file).toLowerCase()
      const immutable =
        extension === '.woff2' ||
        extension === '.woff' ||
        extension === '.ttf' ||
        file.includes(`${sep}assets${sep}`) ||
        file.includes(`${sep}pagefind${sep}`) ||
        file.includes(`${sep}katex${sep}`)

      res.writeHead(200, {
        'content-type': TYPES[extension] ?? 'application/octet-stream',
        'cache-control': immutable ? 'public, max-age=31536000, immutable' : 'no-cache',
      })
      createReadStream(file).pipe(res)
    })()
  })

  /*
   * When the caller does not care which port, pick one from the high range rather than
   * passing 0 and taking whatever the OS offers.
   *
   * Chrome refuses to connect to a set of ports reserved for other protocols — 1719, 2049,
   * 6000 and a couple of dozen more — with ERR_UNSAFE_PORT. An ephemeral port lands on one
   * eventually, which turns into a check that fails a few times a year for no reason anyone
   * can reproduce. Everything above 20000 is safe.
   */
  const listen = (candidate: number): Promise<void> =>
    new Promise((resolve, reject) => {
      server.once('error', reject)
      server.listen(candidate, '127.0.0.1', () => {
        server.removeListener('error', reject)
        resolve()
      })
    })

  if (port !== 0) {
    await listen(port)
  } else {
    let bound = false
    for (let attempt = 0; attempt < 20 && !bound; attempt++) {
      try {
        await listen(20000 + Math.floor(Math.random() * 40000))
        bound = true
      } catch {
        // Port in use; try another.
      }
    }
    if (!bound) throw new Error('could not bind a free port in 20000-60000')
  }

  const address = server.address()
  if (typeof address === 'string' || !address) throw new Error('could not bind a port')

  return {
    url: `http://127.0.0.1:${address.port}${mount}`,
    close: () =>
      new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve()))
      ),
  }
}
