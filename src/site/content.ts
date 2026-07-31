import { readdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'

import { renderNote, type RenderedNote } from '../render/pipeline.ts'
import { detectLanguage, type NoteLanguage } from './language.ts'

/*
 * Discovery: the notes/ directory, read into the tree the whole site is generated from.
 *
 * The folder structure *is* the site structure — there is no manifest, no sidebar config
 * and no required frontmatter. Adding a note means adding a file, which is requirement (5)
 * and the reason none of the ordering rules below are mandatory: each one is an inference
 * with an override, never a thing you must remember to write.
 */

export interface NoteNode {
  kind: 'note'
  /** URL segments, e.g. ['algebra', 'groups']. Empty for the site root. */
  slug: string[]
  title: string
  sortKey: number
  hidden: boolean
  /** Indexed under this language; see src/site/language.ts. */
  lang: NoteLanguage
  rendered: RenderedNote
}

export interface FolderNode {
  kind: 'folder'
  slug: string[]
  title: string
  sortKey: number
  hidden: boolean
  children: TreeNode[]
  /** The folder's own `index.md`, when it has one. */
  index: NoteNode | undefined
}

export type TreeNode = NoteNode | FolderNode

/**
 * A leading number orders a file without appearing in its name: `01-groups.md` sorts first
 * and reads as "Groups". Both `-` and `_` separate, and the prefix may be any digits.
 */
const NUMERIC_PREFIX = /^(\d+)[-_.\s]+/

function splitPrefix(basename: string): { sortKey: number; rest: string } {
  const match = NUMERIC_PREFIX.exec(basename)
  if (!match?.[1]) return { sortKey: Number.POSITIVE_INFINITY, rest: basename }
  return { sortKey: Number(match[1]), rest: basename.slice(match[0].length) }
}

/** `linear-algebra` → `Linear algebra`. Only ever a fallback; an H1 or frontmatter wins. */
function prettify(name: string): string {
  const words = name.replace(/[-_]+/g, ' ').replace(/\s+/g, ' ').trim()
  return words.charAt(0).toUpperCase() + words.slice(1)
}

/**
 * Order: an explicit `order:` or numeric prefix first, then folders, then title.
 *
 * Falling back to folders-before-files only *within* an ordering group means a numeric
 * prefix can interleave them deliberately, while an unordered directory still reads
 * tidily.
 */
function compare(a: TreeNode, b: TreeNode): number {
  if (a.sortKey !== b.sortKey) return a.sortKey - b.sortKey
  if (a.kind !== b.kind) return a.kind === 'folder' ? -1 : 1
  return a.title.localeCompare(b.title)
}

function isMarkdown(name: string): boolean {
  return name.toLowerCase().endsWith('.md')
}

export interface LoadResult {
  root: FolderNode
  /** Every note, flattened — the generator's work list. */
  notes: NoteNode[]
}

async function loadFolder(dir: string, slug: string[]): Promise<FolderNode> {
  const dirents = await readdir(dir, { withFileTypes: true })
  const children: TreeNode[] = []
  let index: NoteNode | undefined

  for (const dirent of dirents) {
    // Dotfiles and underscore-prefixed names are private: somewhere to keep drafts and
    // attachments inside notes/ without them becoming pages.
    if (dirent.name.startsWith('.') || dirent.name.startsWith('_')) continue

    if (dirent.isDirectory()) {
      const child = await loadFolder(join(dir, dirent.name), [...slug, dirent.name])
      // A folder with no notes anywhere beneath it would be an empty page and an empty
      // sidebar branch.
      if (child.children.length > 0 || child.index) children.push(child)
      continue
    }

    if (!dirent.isFile() || !isMarkdown(dirent.name)) continue

    const basename = dirent.name.slice(0, -3)
    const source = await readFile(join(dir, dirent.name), 'utf8')
    const { sortKey, rest } = splitPrefix(basename)

    /*
     * A note becomes a directory — `a/foo.md` is served at `/a/foo/` — so its images, which
     * sit beside the Markdown file, end up one level above the page. An index.md keeps its
     * own directory and needs no adjustment.
     */
    const isIndex = rest.toLowerCase() === 'index' || rest.toLowerCase() === 'readme'
    const rendered = renderNote(source, prettify(rest), { assetPrefix: isIndex ? '' : '../' })
    const node: NoteNode = {
      kind: 'note',
      slug: isIndex ? slug : [...slug, slugify(rest)],
      title: rendered.title,
      sortKey: typeof rendered.frontmatter.order === 'number' ? rendered.frontmatter.order : sortKey,
      hidden: rendered.frontmatter.hidden === true,
      lang: detectLanguage(source),
      rendered,
    }

    if (isIndex) index = node
    else children.push(node)
  }

  children.sort(compare)

  const own = slug.at(-1) ?? ''
  const { sortKey, rest } = splitPrefix(own)
  return {
    kind: 'folder',
    slug,
    // A folder's index.md titles the folder, so renaming a section is done in one place.
    title: index?.title ?? prettify(rest),
    sortKey:
      typeof index?.rendered.frontmatter.order === 'number' ? index.rendered.frontmatter.order : sortKey,
    hidden: index?.hidden ?? false,
    children,
    index,
  }
}

/** URL-safe segment, keeping unicode letters so a CJK filename stays readable in the path. */
export function slugify(name: string): string {
  return name
    .normalize('NFC')
    .replace(/\s+/g, '-')
    .replace(/[/\\?%*:|"<>#]/g, '')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
}

export async function loadNotes(dir: string): Promise<LoadResult> {
  const root = await loadFolder(dir, [])
  const notes: NoteNode[] = []

  const collect = (node: FolderNode): void => {
    if (node.index) notes.push(node.index)
    for (const child of node.children) {
      if (child.kind === 'folder') collect(child)
      else notes.push(child)
    }
  }
  collect(root)

  return { root, notes }
}
