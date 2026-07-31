import type { FolderNode, TreeNode } from '../site/content.ts'

/*
 * The navigation panel.
 *
 * Folders are <details>/<summary>, which is the whole expand-and-collapse mechanism — no
 * JavaScript, and it works before the shell bundle loads or if it never does. The current
 * page's ancestors are written out already `open`, so the tree arrives in the right shape
 * rather than snapping into it after hydration.
 *
 * src/shell/shell.ts adds only persistence on top: remembering which folders you opened
 * elsewhere. That is an enhancement, and its absence costs nothing.
 */

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

/** Path from a page at `depth` back to the site root, e.g. `../../`. */
export function rootPrefix(depth: number): string {
  return depth === 0 ? './' : '../'.repeat(depth)
}

/**
 * Link from the current page to `slug`.
 *
 * Always relative, never absolute: that is what lets the built site work at a domain root,
 * at a project-page subpath like /note/, or opened straight off disk, with no `base`
 * setting and no rebuild.
 */
export function hrefTo(slug: readonly string[], fromDepth: number): string {
  const prefix = rootPrefix(fromDepth)
  return slug.length === 0 ? prefix : `${prefix}${slug.map(encodeURIComponent).join('/')}/`
}

const CHEVRON =
  '<svg class="nav-chevron" viewBox="0 0 16 16" aria-hidden="true" focusable="false">' +
  '<path d="M6 4l4 4-4 4" fill="none" stroke="currentColor" stroke-width="1.75" ' +
  'stroke-linecap="round" stroke-linejoin="round"/></svg>'

function isAncestorOf(folder: readonly string[], page: readonly string[]): boolean {
  return folder.every((segment, i) => page[i] === segment)
}

function renderNode(node: TreeNode, current: readonly string[], depth: number): string {
  if (node.hidden) return ''

  const href = hrefTo(node.slug, depth)
  const isCurrent = node.slug.join('/') === current.join('/')

  if (node.kind === 'note') {
    return (
      `<li class="nav-item">` +
      `<a class="nav-link${isCurrent ? ' is-current' : ''}" href="${escapeHtml(href)}"` +
      `${isCurrent ? ' aria-current="page"' : ''}>${escapeHtml(node.title)}</a>` +
      `</li>`
    )
  }

  const children = node.children.map((child) => renderNode(child, current, depth)).join('')
  if (!children) return ''

  // Open when the current page is inside it — including when it *is* the folder's own page.
  const open = isAncestorOf(node.slug, current)
  const key = escapeHtml(node.slug.join('/'))

  return (
    `<li class="nav-item nav-item--folder">` +
    `<details class="nav-folder"${open ? ' open' : ''} data-path="${key}">` +
    `<summary class="nav-summary">${CHEVRON}` +
    `<a class="nav-link nav-link--folder${isCurrent ? ' is-current' : ''}" href="${escapeHtml(href)}"` +
    `${isCurrent ? ' aria-current="page"' : ''}>${escapeHtml(node.title)}</a>` +
    `</summary>` +
    `<ul class="nav-list">${children}</ul>` +
    `</details>` +
    `</li>`
  )
}

/**
 * Render the whole tree, from the perspective of the page at `current`.
 *
 * The tree is identical on every page apart from which branches are open and which link is
 * marked — cheap to regenerate per page, and it means no client-side routing.
 */
export function renderTree(root: FolderNode, current: readonly string[]): string {
  const depth = current.length
  const items = root.children.map((child) => renderNode(child, current, depth)).join('')
  return `<ul class="nav-list nav-list--root">${items}</ul>`
}
