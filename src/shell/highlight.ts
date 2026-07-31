/*
 * Search-term highlighting, and scrolling to the match rather than to the page.
 *
 * A result link carries `?q=<query>` and the section anchor Pagefind found. The anchor gets
 * the browser to roughly the right place on its own; this module then finds the actual words
 * in the document and takes the reader to the line they matched.
 *
 * Highlighting uses the CSS Custom Highlight API — ranges registered with the browser and
 * painted by a `::highlight()` rule — rather than wrapping matches in <mark>. That is not a
 * stylistic preference. Wrapping would insert elements inside `#write`, which is exactly what
 * doc/dom-contract.md forbids: `kiwi.css` uses `#write > *:first-child`, and a <mark>
 * injected mid-paragraph would also break KaTeX's markup and split text nodes the theme's
 * selectors expect intact. Ranges leave the DOM untouched.
 *
 * Where the API is missing the highlight is simply skipped; the scroll still happens, which
 * is the more useful half.
 */

const HIGHLIGHT_NAME = 'note-search'

/** Subtrees whose text should never be highlighted. */
const SKIP = new Set(['SCRIPT', 'STYLE', 'NOALERT'])

function supported(): boolean {
  return typeof CSS !== 'undefined' && 'highlights' in CSS && typeof Highlight !== 'undefined'
}

/**
 * The words worth highlighting.
 *
 * Pagefind stems, so a query for "rendering" legitimately matches "render" — matching whole
 * words back would highlight nothing on the page you were just sent to. Matching on a prefix
 * of each word is the cheap approximation that keeps the highlight and the result in
 * agreement, and it is why terms shorter than two characters are dropped rather than
 * highlighting a letter everywhere it appears.
 */
function terms(query: string): string[] {
  const words = query
    .toLowerCase()
    .split(/[\s,.;:!?()[\]{}"'`]+/)
    .map((word) => word.replace(/^\W+|\W+$/g, ''))
    .filter((word) => word.length >= 2)
  return [...new Set(words)]
}

/** Every text node under `root`, skipping maths and anything not meant to be read as prose. */
function textNodes(root: Element): Text[] {
  const out: Text[] = []
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = (node as Text).parentElement
      if (!parent) return NodeFilter.FILTER_REJECT
      if (SKIP.has(parent.tagName)) return NodeFilter.FILTER_REJECT
      /*
       * KaTeX emits the formula twice — visually as spans, and again as MathML for assistive
       * technology. Highlighting inside it would paint both copies and cut through glyph
       * markup that is positioned to the pixel.
       */
      if (parent.closest('.katex, .code-tooltip')) return NodeFilter.FILTER_REJECT
      return (node as Text).data.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT
    },
  })
  for (let node = walker.nextNode(); node; node = walker.nextNode()) out.push(node as Text)
  return out
}

function findRanges(root: Element, query: string): Range[] {
  const words = terms(query)
  if (words.length === 0) return []

  const ranges: Range[] = []
  for (const node of textNodes(root)) {
    const haystack = node.data.toLowerCase()
    for (const word of words) {
      let from = haystack.indexOf(word)
      while (from !== -1) {
        const range = document.createRange()
        range.setStart(node, from)
        range.setEnd(node, from + word.length)
        ranges.push(range)
        from = haystack.indexOf(word, from + word.length)
      }
    }
  }
  // Document order, so "first match after the anchor" means what it says.
  ranges.sort((a, b) => {
    const cmp = a.compareBoundaryPoints(Range.START_TO_START, b)
    return cmp
  })
  return ranges
}

function clear(): void {
  if (supported()) CSS.highlights.delete(HIGHLIGHT_NAME)
}

/**
 * Scroll `range` to the middle of the viewport.
 *
 * Centred rather than scrolled just into view: the match is usually mid-paragraph, and
 * landing it at the very top of the window puts it under the sticky bar and leaves none of
 * the sentence that precedes it visible.
 */
function scrollTo(range: Range): void {
  const rect = range.getBoundingClientRect()
  if (rect.height === 0 && rect.width === 0) return
  const target = window.scrollY + rect.top - window.innerHeight / 2
  window.scrollTo({
    top: Math.max(0, target),
    behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
  })
}

export interface ApplyOptions {
  /** Scroll to the first match. Off when the query is being edited on the page you are on. */
  scroll?: boolean
  /** Prefer the first match at or after this element id — Pagefind's section anchor. */
  anchorId?: string | undefined
}

/**
 * Highlight `query` inside the note, and optionally travel to the first match.
 *
 * Returns how many matches were found, so the caller can tell "no matches on this page" from
 * "highlighting is unavailable".
 */
export function applyHighlight(query: string, options: ApplyOptions = {}): number {
  clear()
  const write = document.querySelector('#write')
  if (!write || !query.trim()) return 0

  const ranges = findRanges(write, query)
  if (ranges.length === 0) return 0

  if (supported()) CSS.highlights.set(HIGHLIGHT_NAME, new Highlight(...ranges))

  if (options.scroll) {
    /*
     * The anchor is the section Pagefind matched in. Starting the search for a line from
     * there matters when a common word appears earlier in the page too — otherwise clicking
     * a result about the third section lands you in the first.
     */
    let chosen = ranges[0]
    const anchor = options.anchorId ? document.getElementById(options.anchorId) : null
    if (anchor) {
      const after = ranges.find(
        (range) =>
          anchor.compareDocumentPosition(range.startContainer) & Node.DOCUMENT_POSITION_FOLLOWING
      )
      if (after) chosen = after
    }
    if (chosen) scrollTo(chosen)
  }

  return ranges.length
}

export function clearHighlight(): void {
  clear()
}
