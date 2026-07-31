import './compat.css'
import './bridge.css'
import './shell.css'

import { applyHighlight, clearHighlight } from './highlight.ts'

/*
 * The client shell.
 *
 * Everything here is an enhancement over a page that already works. The tree expands and
 * collapses through <details>, the note is fully rendered, the maths is already typeset and
 * the code already coloured — all of that is in the HTML before a byte of this runs. What
 * this adds is memory (which folders you had open, which palette you chose), the drawer on
 * small screens, and search.
 *
 * Written against the DOM the generator emits, with no framework: the whole thing is
 * smaller than the runtime of any library that could have rendered it.
 */

const root = document.documentElement.dataset['root'] ?? './'

/* ── Theme ─────────────────────────────────────────────────────────────────────────── */

/*
 * No stored choice means no [data-theme] attribute, and the stylesheet's own
 * prefers-color-scheme rule decides — so the site follows the system until the reader
 * overrides it. The first click therefore has to resolve what is *currently* showing,
 * rather than reading an attribute that is deliberately absent.
 */
function effectiveTheme(): 'light' | 'dark' {
  const chosen = document.documentElement.dataset['theme']
  if (chosen === 'light' || chosen === 'dark') return chosen
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

document.querySelector('[data-theme-toggle]')?.addEventListener('click', () => {
  const next = effectiveTheme() === 'dark' ? 'light' : 'dark'
  document.documentElement.dataset['theme'] = next
  try {
    localStorage.setItem('note.theme', next)
  } catch {
    // Private-mode storage failures shouldn't break theming.
  }
})

/* ── Navigation drawer ─────────────────────────────────────────────────────────────── */

const drawerToggle = document.querySelector<HTMLButtonElement>('[data-drawer-toggle]')
const scrim = document.querySelector<HTMLElement>('.drawer-scrim')

function setDrawer(open: boolean): void {
  document.body.dataset['drawer'] = open ? 'open' : 'closed'
  drawerToggle?.setAttribute('aria-expanded', String(open))
  if (scrim) scrim.hidden = !open
}

drawerToggle?.addEventListener('click', () => {
  setDrawer(document.body.dataset['drawer'] !== 'open')
})
scrim?.addEventListener('click', () => setDrawer(false))

/* ── Tree state ────────────────────────────────────────────────────────────────────── */

/*
 * The generator already opens the current page's ancestors, which is the part that matters
 * for orientation. This remembers the rest — a folder you opened to look around stays open
 * as you move through the site.
 *
 * Ancestors of the current page are never closed by this, because the server-rendered
 * `open` attribute is only ever added to, never removed.
 */
const OPEN_KEY = 'note.open-folders'

function readOpen(): Set<string> {
  try {
    const raw = localStorage.getItem(OPEN_KEY)
    return new Set(raw ? (JSON.parse(raw) as string[]) : [])
  } catch {
    return new Set()
  }
}

const openFolders = readOpen()

for (const folder of document.querySelectorAll<HTMLDetailsElement>('.nav-folder')) {
  const path = folder.dataset['path']
  if (!path) continue
  if (openFolders.has(path)) folder.open = true

  folder.addEventListener('toggle', () => {
    if (folder.open) openFolders.add(path)
    else openFolders.delete(path)
    try {
      localStorage.setItem(OPEN_KEY, JSON.stringify([...openFolders]))
    } catch {
      // Non-fatal: the tree still works, it just forgets.
    }
  })
}

/* ── Keeping the reader's place in the tree ────────────────────────────────────────── */

/*
 * Every page is a fresh document, so the panel arrives scrolled to the top — and with a few
 * hundred notes that means clicking one near the bottom of the tree scrolls the tree away
 * from the note you just opened, which is exactly when you least want to lose it.
 *
 * The position is stored per tab rather than per page: it is the same tree everywhere, and
 * what the reader is keeping their place in is the tree, not this page's copy of it.
 *
 * Restored at module scope, not on DOMContentLoaded. This bundle is a deferred module, so it
 * runs after parsing and before the first paint — late enough that `.sidebar` exists, early
 * enough that nobody sees it at the top first. Waiting for an event would put the correction
 * after a frame had already been shown, and the jump is the whole thing being avoided.
 */
const sidebar = document.querySelector<HTMLElement>('.sidebar')
const SCROLL_KEY = 'note.tree-scroll'

if (sidebar) {
  try {
    const saved = sessionStorage.getItem(SCROLL_KEY)
    if (saved !== null) sidebar.scrollTop = Number(saved)
  } catch {
    // Private-mode storage failures shouldn't break navigation.
  }

  /*
   * Then make sure the current note is actually on screen. This is what covers arriving from
   * outside the tree — a shared link, a search result, a fresh tab — where there is no stored
   * position to restore and the note may sit hundreds of pixels down. Only nudges when the
   * link is out of view, so a restored position is left exactly as the reader left it.
   */
  const current = sidebar.querySelector<HTMLElement>('.nav-link.is-current')
  if (current) {
    const link = current.getBoundingClientRect()
    const pane = sidebar.getBoundingClientRect()
    if (link.top < pane.top || link.bottom > pane.bottom) {
      sidebar.scrollTop += link.top - pane.top - pane.height / 2 + link.height / 2
    }
  }

  const remember = (): void => {
    try {
      sessionStorage.setItem(SCROLL_KEY, String(sidebar.scrollTop))
    } catch {
      // Non-fatal: the tree still works, it just forgets.
    }
  }

  /*
   * `pagehide` is what actually captures the position on the way out — it fires on
   * navigation, including back and forward, and unlike `beforeunload` it does not disable
   * the back/forward cache. The throttled scroll handler is the backstop for the cases it
   * misses, such as a tab discarded in the background.
   */
  window.addEventListener('pagehide', remember)

  let scrollTimer: number | undefined
  sidebar.addEventListener(
    'scroll',
    () => {
      window.clearTimeout(scrollTimer)
      scrollTimer = window.setTimeout(remember, 150)
    },
    { passive: true }
  )
}

/* ── Contents highlighting ─────────────────────────────────────────────────────────── */

/*
 * Marks the section you are reading. The observer watches the headings themselves and keeps
 * the last one to have crossed the top of the viewport, which is what makes the highlight
 * track the section rather than flicker between two headings that are both on screen.
 */
const tocLinks = new Map<string, HTMLElement>()
for (const item of document.querySelectorAll<HTMLElement>('.toc .md-toc-item')) {
  const ref = item.dataset['ref']
  if (ref) tocLinks.set(ref, item)
}

if (tocLinks.size > 0) {
  const seen = new Map<string, boolean>()

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) seen.set(entry.target.id, entry.isIntersecting)

      let active: string | undefined
      for (const id of tocLinks.keys()) {
        const heading = document.getElementById(id)
        if (!heading) continue
        if (seen.get(id) || heading.getBoundingClientRect().top < 120) active = id
        if (seen.get(id)) break
      }

      for (const [id, item] of tocLinks) item.classList.toggle('is-active', id === active)
    },
    { rootMargin: '-100px 0px -70% 0px', threshold: 0 }
  )

  for (const id of tocLinks.keys()) {
    const heading = document.getElementById(id)
    if (heading) observer.observe(heading)
  }
}

/* ── Search ────────────────────────────────────────────────────────────────────────── */

interface PagefindSubResult {
  title: string
  /** Site-absolute, and already carrying the section's `#anchor`. */
  url: string
  excerpt: string
}

interface PagefindResultData {
  url: string
  excerpt: string
  meta: Record<string, string | undefined>
  /** One per section of the page that matched, in relevance order. */
  sub_results?: PagefindSubResult[]
}

interface PagefindApi {
  search(query: string): Promise<{ results: Array<{ data(): Promise<PagefindResultData> }> }>
  options(config: Record<string, unknown>): Promise<void>
  mergeIndex(bundlePath: string, config: Record<string, unknown>): Promise<void>
}

const dialog = document.querySelector<HTMLDialogElement>('.search-dialog')
const input = document.querySelector<HTMLInputElement>('[data-search-input]')
const resultList = document.querySelector<HTMLElement>('[data-search-results]')
const emptyNote = document.querySelector<HTMLElement>('[data-search-empty]')

let pagefind: PagefindApi | undefined
let loadFailed = false
let activeIndex = 0
let generation = 0

/*
 * Pagefind's bundle is generated by its CLI *after* Vite has run, so Vite must not try to
 * resolve this path — hence the ignore comment. Loading it on first use rather than at
 * startup keeps it off the critical path of every page view; most visits never search.
 *
 * The URL is resolved against `document.baseURI`, and that is load-bearing rather than
 * tidiness. A relative specifier in `import()` resolves against the *module* — this bundle,
 * under /assets/ — not against the page. `./pagefind/pagefind.js` therefore asks for
 * /assets/pagefind/, which does not exist. It fails only on some pages, because from a note
 * two levels deep the prefix is `../../`, which climbs out of /assets/ and happens to land
 * in the right place.
 */
async function loadPagefind(): Promise<PagefindApi | undefined> {
  if (pagefind || loadFailed) return pagefind
  try {
    const url = new URL(`${root}pagefind/pagefind.js`, document.baseURI).href
    const module = (await import(/* @vite-ignore */ url)) as PagefindApi
    /*
     * `baseUrl: '/'` pins result URLs to the form they were indexed in. Left alone, Pagefind
     * infers a base from where its own bundle was loaded, so on a project-page deploy every
     * result comes back already carrying `/note/` — which `resolve()` then prefixes again,
     * producing /note/note/…. Rooting them here keeps exactly one place that knows about
     * deploy paths, which is `resolve()`.
     */
    await module.options({ excerptLength: 24, baseUrl: '/' })

    /*
     * Reach the notes written in the other language.
     *
     * Pagefind builds one index per language, because tokenising is language-specific — and
     * that difference is not cosmetic. English splits on whitespace, which is useless for
     * Chinese: `N皇后` has no spaces in it, so under the English tokeniser it is a single
     * token and searching `皇后` matches nothing. Chinese needs a segmenter, which Pagefind's
     * Extended build applies only to content marked as Chinese.
     *
     * Marking each note correctly fixes the indexing, and creates a second problem: the
     * runtime loads only the index matching the page you happen to be standing on, so a
     * search from an English note could not see the Chinese ones. Merging the rest makes the
     * search cover the whole site from anywhere.
     *
     * The query itself is still tokenised by *this* page's language, so an English word
     * searched from an English page also gets stemming, and from a Chinese page does not.
     * Both find the note; one ranks a little wider.
     */
    const languages = (document.documentElement.dataset['langs'] ?? '').split(/\s+/).filter(Boolean)
    const here = document.documentElement.lang
    for (const language of languages) {
      if (language === here) continue
      await module.mergeIndex(new URL(`${root}pagefind/`, document.baseURI).href, {
        language,
        baseUrl: '/',
      })
    }

    pagefind = module
  } catch {
    // Expected in `vite dev`, where the index has not been built yet.
    loadFailed = true
    if (emptyNote) emptyNote.textContent = 'Search index unavailable — run a full build.'
  }
  return pagefind
}

function setEmpty(message: string): void {
  if (resultList) resultList.innerHTML = ''
  if (emptyNote) {
    emptyNote.textContent = message
    emptyNote.hidden = false
  }
}

/**
 * Turn a Pagefind URL into a link that lands on the matched line.
 *
 * Pagefind indexes dist/, so its URLs are site-absolute and have to be re-rooted onto our
 * relative scheme. On top of that the query rides along as `?q=`, which is what lets the
 * destination page highlight the term and scroll to it — the `#anchor` alone only reaches
 * the section.
 *
 * The query goes before the fragment because that is where a query string belongs; putting
 * it after would make it part of the fragment and invisible to `location.search`.
 */
function resolve(url: string, query: string): string {
  const [path = '', hash] = url.replace(/^\//, '').split('#')
  const search = `?q=${encodeURIComponent(query)}`
  return `${root}${path}${search}${hash ? `#${hash}` : ''}`
}

function renderResults(items: PagefindResultData[], query: string): void {
  if (!resultList || !emptyNote) return
  if (items.length === 0) {
    setEmpty('No matches.')
    return
  }

  emptyNote.hidden = true
  activeIndex = 0
  resultList.innerHTML = items
    .map((item, index) => {
      const title = item.meta['title'] ?? 'Untitled'
      /*
       * Prefer the section Pagefind matched in over the page as a whole: its URL carries the
       * heading anchor, so the reader arrives at the right part of a long note instead of at
       * the top of it.
       */
      const section = item.sub_results?.[0]
      const href = resolve(section?.url ?? item.url, query)
      const page = decodeURIComponent(item.url.replace(/^\/|\/$/g, '')) || 'Home'
      const crumb = section && section.title !== title ? `${page} › ${section.title}` : page
      return (
        `<li class="search-result${index === 0 ? ' is-active' : ''}">` +
        `<a class="search-result__link" href="${escapeAttr(href)}">` +
        `<span class="search-result__crumb">${escapeHtml(crumb)}</span>` +
        `<span class="search-result__title">${escapeHtml(title)}</span>` +
        // Pagefind's excerpt carries its own <mark> tags around the hit, which is the only
        // HTML we accept from it.
        `<span class="search-result__excerpt">${section?.excerpt ?? item.excerpt}</span>` +
        `</a></li>`
      )
    })
    .join('')
}

function escapeHtml(value: string): string {
  const el = document.createElement('span')
  el.textContent = value
  return el.innerHTML
}

function escapeAttr(value: string): string {
  return value.replace(/&/g, '&amp;').replace(/"/g, '&quot;')
}

async function runSearch(query: string): Promise<void> {
  const mine = ++generation

  /*
   * The highlight follows the search box, not the page load: type and the current note
   * updates, clear the box and it goes. Without scrolling, because the reader is already
   * reading — moving the page under a search they have not committed to would be hostile.
   */
  syncQueryParam(query)
  if (query.trim()) applyHighlight(query)
  else clearHighlight()

  if (!query.trim()) {
    setEmpty('Type to search.')
    return
  }

  const api = await loadPagefind()
  if (!api || mine !== generation) return

  const search = await api.search(query)
  if (mine !== generation) return // a later keystroke already won

  const items = await Promise.all(search.results.slice(0, 12).map((result) => result.data()))
  if (mine !== generation) return
  renderResults(items, query)
}

/**
 * Keep `?q=` in the address bar in step with the search box.
 *
 * So that reloading, or sharing the URL, reproduces what is on screen — and so that clearing
 * the box does not leave a stale query behind to reappear on the next reload.
 * `replaceState` rather than `pushState`: typing a query is not a navigation, and every
 * keystroke becoming a back-button step would trap the reader.
 */
function syncQueryParam(query: string): void {
  const url = new URL(window.location.href)
  if (query.trim()) url.searchParams.set('q', query)
  else url.searchParams.delete('q')
  if (url.href !== window.location.href) window.history.replaceState(null, '', url)
}

function moveActive(delta: number): void {
  const items = [...document.querySelectorAll<HTMLElement>('.search-result')]
  if (items.length === 0) return
  items[activeIndex]?.classList.remove('is-active')
  activeIndex = (activeIndex + delta + items.length) % items.length
  const next = items[activeIndex]
  next?.classList.add('is-active')
  next?.scrollIntoView({ block: 'nearest' })
}

function openSearch(): void {
  if (!dialog) return
  dialog.showModal()
  input?.focus()
  input?.select()
  void loadPagefind()
  // Arriving from a result leaves the term in the box; show its results again rather than an
  // empty panel under a filled-in query.
  if (input?.value.trim() && !resultList?.children.length) void runSearch(input.value)
}

document.querySelector('[data-search-open]')?.addEventListener('click', openSearch)

let debounce: number | undefined
input?.addEventListener('input', () => {
  window.clearTimeout(debounce)
  debounce = window.setTimeout(() => void runSearch(input.value), 120)
})

input?.addEventListener('keydown', (event) => {
  if (event.key === 'ArrowDown') {
    event.preventDefault()
    moveActive(1)
  } else if (event.key === 'ArrowUp') {
    event.preventDefault()
    moveActive(-1)
  } else if (event.key === 'Enter') {
    event.preventDefault()
    const link = document.querySelectorAll<HTMLAnchorElement>('.search-result__link')[activeIndex]
    if (link) window.location.href = link.href
  }
})

document.addEventListener('keydown', (event) => {
  const target = event.target as HTMLElement | null
  const typing =
    target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA' || target?.isContentEditable

  if ((event.key === 'k' || event.key === 'K') && (event.metaKey || event.ctrlKey)) {
    event.preventDefault()
    openSearch()
  } else if (event.key === '/' && !typing && !dialog?.open) {
    event.preventDefault()
    openSearch()
  } else if (event.key === 'Escape' && document.body.dataset['drawer'] === 'open') {
    setDrawer(false)
  }
})

// Clicking the backdrop closes the dialog; clicking the panel must not.
dialog?.addEventListener('click', (event) => {
  if (event.target === dialog) dialog.close()
})

/* ── Arriving from a search result ─────────────────────────────────────────────────── */

/*
 * A result link carries `?q=` and the section's `#anchor`. The browser handles the anchor on
 * its own, which gets the reader to the right heading; this then finds the words themselves
 * and moves to the line that actually matched.
 *
 * The query is put back into the search box too, so it is still there when the reader opens
 * search again — and so clearing it is what removes the highlight, which is the only way to
 * remove it. That symmetry is the whole behaviour: the highlight is on screen for exactly as
 * long as the term is in the box.
 */
{
  const initial = new URL(window.location.href).searchParams.get('q')
  if (initial?.trim()) {
    if (input) input.value = initial
    const anchorId = window.location.hash.slice(1) || undefined

    // After layout, or the range rectangles used to pick a scroll target are all zero. Fonts
    // matter as much as layout here: LXGW WenKai is a CJK family loaded in subsets, and text
    // reflows when it lands, which would leave the match a few lines off.
    const go = (): void => {
      applyHighlight(initial, { scroll: true, anchorId })
    }
    if (document.fonts?.status === 'loaded') requestAnimationFrame(go)
    else void (document.fonts?.ready ?? Promise.resolve()).then(() => requestAnimationFrame(go))
  }
}
