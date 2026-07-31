import { cp, mkdir, readdir, readFile, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { loadNotes, type FolderNode, type NoteNode, type TreeNode } from '../src/site/content.ts'
import { latinSubsets } from '../src/site/fonts.ts'
import { buildTheme } from '../src/site/theme.ts'
import { renderPage } from '../src/shell/page.ts'
import { renderToc } from '../src/shell/toc.ts'
import { hrefTo, renderTree, rootPrefix } from '../src/shell/tree.ts'
import { siteConfig } from '../site.config.ts'

/*
 * Phase two of the build: Vite has produced the shell bundle, and this turns notes/ into
 * dist/.
 *
 * Every page is written whole — nav tree, contents, note — because the tree differs per page
 * only in which branches are open and which link is marked, and regenerating it is far
 * cheaper than shipping a router to avoid doing so.
 */

const root = fileURLToPath(new URL('..', import.meta.url))
const dist = join(root, 'dist')
const notesDir = join(root, 'notes')

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

/** Read the hashed asset names Vite wrote, so pages can link them. */
async function readAssets(): Promise<{ js: string; css: string }> {
  const manifestPath = join(dist, '.vite/manifest.json')
  if (!existsSync(manifestPath)) {
    throw new Error(
      'dist/.vite/manifest.json is missing — run `npm run build:assets` first, or just ' +
        '`npm run build`, which runs the phases in order.'
    )
  }
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8')) as Record<
    string,
    { file: string; css?: string[] }
  >
  const entry = manifest['src/shell/shell.ts']
  if (!entry?.file || !entry.css?.[0]) {
    throw new Error('Vite manifest has no shell entry with CSS; check vite.config.ts input.')
  }
  return { js: entry.file, css: entry.css[0] }
}

/**
 * The page for a folder that has no index.md.
 *
 * A directory is a URL whether or not anyone wrote a page for it, so it gets a real one
 * rather than a 404 — otherwise clicking a folder in the sidebar is a dead end.
 */
function listingBody(folder: FolderNode, depth: number): string {
  const visible = folder.children.filter((child) => !child.hidden)
  const items = visible
    .map((child: TreeNode) => {
      const count =
        child.kind === 'folder'
          ? `${child.children.length} item${child.children.length === 1 ? '' : 's'}`
          : 'Note'
      return (
        `<li><a class="listing__link" href="${escapeHtml(hrefTo(child.slug, depth))}">` +
        `<span class="listing__title">${escapeHtml(child.title)}</span><br>` +
        `<span class="listing__kind">${count}</span></a></li>`
      )
    })
    .join('')

  return (
    `<div id="write" class="typora-editor"${pagefindAttrs(folder.title)}>` +
    `<h1 class="md-end-block md-heading">${escapeHtml(folder.title)}</h1>` +
    `<ul class="listing">${items}</ul>` +
    `</div>`
  )
}

/*
 * Confine the search index to the note.
 *
 * Without `data-pagefind-body` Pagefind indexes every <body> wholesale, which means the
 * navigation panel — the same tree on all pages — is indexed once per page. Every query
 * matching a note's *title* would then hit every page in the site.
 *
 * The title is given explicitly rather than left to Pagefind's own <h1> heuristic, because a
 * note whose first heading is not its title (or which has none) would otherwise show up in
 * results as "Untitled".
 */
function pagefindAttrs(title: string): string {
  return ` data-pagefind-body data-pagefind-meta="title:${escapeHtml(title)}"`
}

/**
 * Mirror every non-Markdown file from `notes/` into `dist/`.
 *
 * Dotfiles are skipped, matching discovery. Underscore-prefixed names are *not*: that rule
 * exists so a draft does not become a page, and an attachment folder named `_img` is still
 * something a published note may point at.
 */
async function copyAssets(from: string, to: string): Promise<number> {
  let copied = 0

  async function walk(dir: string, out: string): Promise<void> {
    for (const dirent of await readdir(dir, { withFileTypes: true })) {
      if (dirent.name.startsWith('.')) continue
      const source = join(dir, dirent.name)

      if (dirent.isDirectory()) {
        await walk(source, join(out, dirent.name))
        continue
      }
      if (!dirent.isFile() || dirent.name.toLowerCase().endsWith('.md')) continue

      await mkdir(out, { recursive: true })
      await cp(source, join(out, dirent.name))
      copied += 1
    }
  }

  await walk(from, to)
  return copied
}

interface Page {
  slug: string[]
  title: string
  description: string
  body: string
  toc: string
  hasMath: boolean
  lang: string
}

function excerpt(text: string): string {
  const trimmed = text.slice(0, 180).trim()
  return trimmed.length < text.length ? `${trimmed}…` : trimmed
}

function pageFor(node: TreeNode): Page {
  if (node.kind === 'note') {
    return {
      slug: node.slug,
      title: node.title,
      description: excerpt(node.rendered.text),
      body: `<div id="write" class="typora-editor"${pagefindAttrs(node.title)}>\n${node.rendered.html}</div>`,
      toc: renderToc(node.rendered.headings),
      hasMath: node.rendered.html.includes('katex'),
      lang: node.lang,
    }
  }

  if (node.index) return { ...pageFor(node.index), slug: node.slug, title: node.title }

  return {
    slug: node.slug,
    title: node.title,
    description: `Notes in ${node.title}.`,
    body: listingBody(node, node.slug.length),
    toc: '',
    hasMath: false,
    // A generated listing is just note titles; index it as whatever its own notes mostly are.
    lang: node.children.filter((c) => c.kind === 'note' && c.lang === 'zh').length * 2 >
      node.children.length
      ? 'zh'
      : 'en',
  }
}

function collectPages(folder: FolderNode, out: Page[]): void {
  out.push(pageFor(folder))
  for (const child of folder.children) {
    if (child.kind === 'folder') collectPages(child, out)
    else out.push(pageFor(child))
  }
}

async function main(): Promise<void> {
  if (!existsSync(notesDir)) {
    throw new Error(`notes/ not found at ${notesDir} — that is where the Markdown lives.`)
  }

  const assets = await readAssets()
  const { root: tree, notes } = await loadNotes(notesDir)

  /*
   * Everything in notes/ that is not Markdown, copied across with its directory structure
   * intact: images, and whatever else a note links to.
   *
   * Typora keeps a note's images in a sibling `<name>.assets/` folder and references them
   * relatively, so preserving the layout is what makes those references resolve. The `../`
   * that reconciles it with a note becoming a directory is applied when rendering — see
   * TransformOptions in src/render/typora-hast.ts.
   */
  const copiedAssets = await copyAssets(notesDir, dist)

  // The single stylesheet: Typora's base, the theme, and both palettes.
  await writeFile(join(dist, 'theme.css'), await buildTheme(root), 'utf8')

  /*
   * KaTeX's stylesheet and glyph fonts, copied straight out of the installed package rather
   * than kept in src/styles/: they are a plain dependency, unmodified, and tracking the
   * version we actually resolve is better than a stale copy.
   */
  const katexDist = join(root, 'node_modules/katex/dist')
  await mkdir(join(dist, 'katex'), { recursive: true })
  await cp(join(katexDist, 'katex.min.css'), join(dist, 'katex/katex.min.css'))
  await cp(join(katexDist, 'fonts'), join(dist, 'katex/fonts'), { recursive: true })

  /*
   * Computed once: the subsets are the same on every page, and the stylesheet does not change
   * between them.
   */
  const fontPreloads = await latinSubsets(
    join(root, 'public/fonts/lxgw-wenkai/lxgw-wenkai.css')
  )

  const pages: Page[] = []
  collectPages(tree, pages)

  /*
   * Pagefind builds one index per language, and its runtime loads only the one matching the
   * page it is on. Publishing the full set lets the client merge the others, so a search from
   * an English note still reaches the Chinese ones — see loadPagefind in src/shell/shell.ts.
   */
  const languages = [...new Set(pages.map((page) => page.lang))].sort()

  for (const page of pages) {
    const depth = page.slug.length
    const html = renderPage({
      title: page.title,
      siteTitle: siteConfig.title,
      description: page.description || siteConfig.description,
      root: rootPrefix(depth),
      assets,
      nav: renderTree(tree, page.slug),
      toc: page.toc,
      body: page.body,
      hasMath: page.hasMath,
      fontPreloads,
      lang: page.lang,
      languages,
      editUrl: siteConfig.editBaseUrl
        ? `${siteConfig.editBaseUrl}/${['notes', ...page.slug].join('/')}.md`
        : undefined,
    })

    const dir = join(dist, ...page.slug)
    await mkdir(dir, { recursive: true })
    await writeFile(join(dir, 'index.html'), html, 'utf8')
  }

  // Warnings last, so they are the final thing on screen rather than scrolled past.
  const mathFailures = notes.flatMap((note: NoteNode) =>
    note.rendered.mathFailures.map((failure) => ({ note, failure }))
  )
  for (const { note, failure } of mathFailures) {
    console.warn(`  formula in /${note.slug.join('/')}: ${failure.message}`)
  }

  const unknown = new Set(notes.flatMap((note: NoteNode) => note.rendered.unknownLanguages))
  if (unknown.size > 0) {
    console.warn(`  no grammar for code fences tagged: ${[...unknown].sort().join(', ')}`)
  }

  /*
   * Links into the author's own machine — `C:\Users\…`, `file:///…`. Typora writes these
   * when its image folder is configured globally rather than per-document, and the files are
   * nowhere near the repository, so there is nothing to copy and nothing to rewrite. Listed
   * per note, because the fix is to move the image beside its Markdown and re-link it.
   */
  const unresolvable = notes.filter((note: NoteNode) => note.rendered.unresolvableAssets.length > 0)
  if (unresolvable.length > 0) {
    const total = unresolvable.reduce((n, note) => n + note.rendered.unresolvableAssets.length, 0)
    console.warn(
      `
  ${total} link(s) in ${unresolvable.length} note(s) point outside the repository ` +
        `and will be broken:`
    )
    for (const note of unresolvable.slice(0, 8)) {
      console.warn(`    /${note.slug.join('/')} -> ${note.rendered.unresolvableAssets[0]}`)
    }
    if (unresolvable.length > 8) console.warn(`    … and ${unresolvable.length - 8} more note(s)`)
  }

  console.log(
    `
${pages.length} pages from ${notes.length} notes, ${copiedAssets} assets` +
      (mathFailures.length > 0 ? `, ${mathFailures.length} formula warning(s)` : '')
  )
}

await main()
