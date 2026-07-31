/*
 * The HTML document.
 *
 * Two things here are load-bearing beyond the markup itself.
 *
 * Every URL is relative, computed from the page's own depth. That is what lets one build
 * work at a domain root, at a project-page subpath like /note/, and opened straight off
 * disk — no `base` setting, no rebuild, and nothing to get wrong when the repository is
 * renamed. `document.documentElement.dataset.root` hands the same prefix to the client so
 * it can reach Pagefind's index.
 *
 * The stylesheet order establishes the cascade: dist/theme.css declares
 * `@layer typora-base, typora-theme, typora-bridge, app` before anything else arrives, so
 * the shell bundle's own layers only ever fill slots that already exist in the right order.
 */

export interface PageAssets {
  js: string
  css: string
}

export interface PageOptions {
  title: string
  siteTitle: string
  description: string
  /** `./`, `../`, `../../` … back to the site root. */
  root: string
  assets: PageAssets
  /** Sidebar markup from src/shell/tree.ts. */
  nav: string
  /** Typora `.md-toc` markup from src/shell/toc.ts; empty when the note is too short. */
  toc: string
  /** The rendered note: a full `<div id="write">`. */
  body: string
  /** Skip the KaTeX stylesheet on pages with no formula — it is ~23 KB plus glyph fonts. */
  hasMath: boolean
  /** Webfont subsets to fetch alongside the stylesheet; see src/site/fonts.ts. */
  fontPreloads: readonly string[]
  /** Drives <html lang>, which is what decides how Pagefind tokenises this page. */
  lang: string
  /** Every language the site was indexed in, so the client can merge the other indexes. */
  languages: readonly string[]
  /** Breadcrumb-ish trail shown under the title in search results and the footer. */
  editUrl?: string | undefined
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

const ICON_SEARCH =
  '<svg viewBox="0 0 20 20" aria-hidden="true" focusable="false"><circle cx="9" cy="9" r="6" ' +
  'fill="none" stroke="currentColor" stroke-width="1.7"/><path d="M13.5 13.5 17 17" ' +
  'stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>'

const ICON_MENU =
  '<svg viewBox="0 0 20 20" aria-hidden="true" focusable="false"><path d="M3 5h14M3 10h14M3 15h14" ' +
  'stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>'

const ICON_THEME =
  '<svg class="icon-button__moon" viewBox="0 0 20 20" aria-hidden="true" focusable="false">' +
  '<path d="M16 12.3A6.8 6.8 0 0 1 7.7 4a6.8 6.8 0 1 0 8.3 8.3Z" fill="none" stroke="currentColor" ' +
  'stroke-width="1.6" stroke-linejoin="round"/></svg>' +
  '<svg class="icon-button__sun" viewBox="0 0 20 20" aria-hidden="true" focusable="false">' +
  '<circle cx="10" cy="10" r="3.6" fill="none" stroke="currentColor" stroke-width="1.6"/>' +
  '<path d="M10 1.5v2M10 16.5v2M18.5 10h-2M3.5 10h-2M15.9 4.1l-1.4 1.4M5.5 14.5l-1.4 1.4' +
  'M15.9 15.9l-1.4-1.4M5.5 5.5 4.1 4.1" stroke="currentColor" stroke-width="1.6" ' +
  'stroke-linecap="round"/></svg>'

/*
 * Expand all / collapse all for the navigation tree. Diverging chevrons for expand,
 * converging for collapse — the branches opening outward and folding back in.
 */
const ICON_EXPAND =
  '<svg viewBox="0 0 20 20" aria-hidden="true" focusable="false">' +
  '<path d="M8.5 5 3.5 10l5 5" fill="none" stroke="currentColor" stroke-width="1.7" ' +
  'stroke-linecap="round" stroke-linejoin="round"/>' +
  '<path d="M11.5 5l5 5-5 5" fill="none" stroke="currentColor" stroke-width="1.7" ' +
  'stroke-linecap="round" stroke-linejoin="round"/></svg>'

const ICON_COLLAPSE =
  '<svg viewBox="0 0 20 20" aria-hidden="true" focusable="false">' +
  '<path d="M3.5 5l5 5-5 5" fill="none" stroke="currentColor" stroke-width="1.7" ' +
  'stroke-linecap="round" stroke-linejoin="round"/>' +
  '<path d="M16.5 5l-5 5 5 5" fill="none" stroke="currentColor" stroke-width="1.7" ' +
  'stroke-linecap="round" stroke-linejoin="round"/></svg>'

/*
 * Applied before first paint, so a reader who has chosen a palette never sees the other one
 * flash. Deliberately does nothing when no choice has been stored: with no [data-theme]
 * attribute the stylesheet's own `prefers-color-scheme` rule decides, which means the site
 * keeps following the system setting — including a change made while the page is open —
 * until the reader overrides it explicitly.
 */
const THEME_BOOTSTRAP =
  `try{var t=localStorage.getItem("note.theme");` +
  `if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`

export function renderPage(options: PageOptions): string {
  const { root, assets } = options
  const title =
    options.title === options.siteTitle
      ? options.siteTitle
      : `${options.title} · ${options.siteTitle}`

  return `<!doctype html>
<html lang="${escapeHtml(options.lang)}" data-root="${escapeHtml(root)}"
      data-langs="${escapeHtml(options.languages.join(' '))}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(title)}</title>
<meta name="description" content="${escapeHtml(options.description)}">
<meta name="color-scheme" content="light dark">
<script>${THEME_BOOTSTRAP}</script>
<link rel="icon" href="${root}favicon.svg" type="image/svg+xml">${options.fontPreloads
    .map(
      (file) =>
        // Ahead of the stylesheet that references them, so the fetch starts immediately
        // rather than after the CSS is parsed and the text laid out — which is the whole
        // point, and the difference between a flash of the fallback face and none.
        //
        // `crossorigin` is required even same-origin: font fetches are always CORS-mode, and
        // without it the preload is discarded and the file fetched a second time.
        `\n<link rel="preload" as="font" type="font/woff2" crossorigin ` +
        `href="${root}fonts/lxgw-wenkai/${file}">`
    )
    .join('')}
<link rel="stylesheet" href="${root}fonts/lxgw-wenkai/lxgw-wenkai.css">
<link rel="stylesheet" href="${root}theme.css">${
    options.hasMath ? `\n<link rel="stylesheet" href="${root}katex/katex.min.css">` : ''
  }
<link rel="stylesheet" href="${root}${assets.css}">
<script type="module" src="${root}${assets.js}"></script>
</head>
<body>
<a class="skip-link" href="#write">Skip to content</a>
<div class="drawer-scrim" data-drawer-close hidden></div>

<header class="topbar pane">
  <button class="icon-button topbar__menu" type="button" data-drawer-toggle
          aria-label="Show navigation" aria-expanded="false" aria-controls="sidebar">${ICON_MENU}</button>
  <a class="topbar__home" href="${root}">${escapeHtml(options.siteTitle)}</a>
  <div class="topbar__spacer"></div>
  <button class="search-trigger" type="button" data-search-open aria-label="Search notes">
    ${ICON_SEARCH}<span class="search-trigger__label">Search</span>
    <kbd class="search-trigger__key">Ctrl K</kbd>
  </button>
  <button class="icon-button" type="button" data-theme-toggle
          aria-label="Switch between light and dark">${ICON_THEME}</button>
</header>

<div class="layout">
  <aside class="sidebar pane" id="sidebar" aria-label="Notes">
    <div class="sidebar__toolbar">
      <span class="sidebar__heading">Notes</span>
      <button class="icon-button" type="button" data-tree-expand
              aria-label="Expand all folders">${ICON_EXPAND}</button>
      <button class="icon-button" type="button" data-tree-collapse
              aria-label="Collapse all folders">${ICON_COLLAPSE}</button>
    </div>
    ${options.nav}
  </aside>

  <main class="main">
    ${options.body}
    ${options.editUrl ? `<footer class="note-footer"><a href="${escapeHtml(options.editUrl)}">Edit this page</a></footer>` : ''}
  </main>

  ${
    options.toc
      ? `<aside class="toc pane" aria-label="On this page">
    <p class="toc__heading">On this page</p>
    ${options.toc}
  </aside>`
      : '<div></div>'
  }
</div>

<dialog class="search-dialog" aria-label="Search notes">
  <form class="search-field" method="dialog" onsubmit="return false">
    ${ICON_SEARCH}
    <input class="search-input" type="search" placeholder="Search notes…" autocomplete="off"
           spellcheck="false" aria-label="Search query" data-search-input>
  </form>
  <ul class="search-results" data-search-results></ul>
  <p class="search-empty" data-search-empty>Type to search.</p>
</dialog>
</body>
</html>
`
}
