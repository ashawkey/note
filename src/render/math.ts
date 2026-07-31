import katex from 'katex'

/*
 * Formulas, rendered once at build time.
 *
 * An editor has to load KaTeX lazily, because it cannot know whether the document being
 * typed will ever contain a formula. A generator knows exactly, and knows it before the
 * reader does — so the maths is rendered here and the browser is never asked to load KaTeX
 * at all. Only katex.min.css and the glyph fonts ship, and a page with no formula pulls
 * neither.
 *
 * `.md-inline-math` and `.md-math-block` are the containers the stylesheets position maths
 * through (doc/dom-contract.md). They are emitted even though we render with KaTeX rather
 * than Typora's MathJax: what is lost is glyph-level `.MathJax` styling, while placement —
 * which is what the container classes control — still works.
 */

export interface MathFailure {
  tex: string
  message: string
}

const OPTIONS = {
  strict: false as const,
  // KaTeX emits both HTML and MathML. The MathML half is what makes a formula selectable,
  // copyable and legible to a screen reader — and it is the text Pagefind indexes, so a
  // formula is searchable by its notation rather than being a hole in the index.
  output: 'htmlAndMathml' as const,
  trust: false,
  errorColor: '#a2331f',
}

/**
 * Render TeX to KaTeX's HTML, collecting failures rather than aborting.
 *
 * One malformed formula should not fail a build of every note, so a failure falls back to
 * KaTeX's own error rendering — the offending source, in red, in place. That is easy to
 * miss in a long document, which is why the message is also pushed onto `failures` for the
 * generator to report at the end.
 */
export function renderMath(tex: string, displayMode: boolean, failures: MathFailure[]): string {
  try {
    return katex.renderToString(tex, { ...OPTIONS, displayMode, throwOnError: true })
  } catch (error) {
    failures.push({
      tex,
      message: error instanceof Error ? error.message : String(error),
    })
    return katex.renderToString(tex, { ...OPTIONS, displayMode, throwOnError: false })
  }
}
