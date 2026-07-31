import type { ElementContent, RootContent } from 'hast'

import { refractor, resolveLanguage } from './languages.ts'
import { toCodeMirrorClass } from './token-map.ts'

/*
 * Syntax highlighting for fenced code, in CodeMirror's vocabulary.
 *
 * Typora highlights fences with CodeMirror 5, so themes colour code through
 * `.cm-s-inner .cm-keyword` and friends — the largest proprietary surface in the whole
 * styled surface in kiwi.css (doc/dom-contract.md). Emitting Prism's own `token keyword`
 * names instead would leave every one of those selectors resolving but matching nothing, and
 * code would render in a single flat colour: a failure the DOM assertions cannot see,
 * because the classes are all present and correct.
 *
 * So we tokenise with Prism and relabel through ./token-map.ts. Highlighting runs once per
 * fence at build time, which is why there is no cache here and no incremental path — both
 * only earn their keep when the same block is re-highlighted on every keystroke.
 */

interface TokenRun {
  text: string
  chain: string[]
}

/**
 * Flatten Prism's nested token tree into a linear list of text runs, each carrying the
 * class chain of every element enclosing it.
 *
 * The nesting is real and deep — `php > language-php > string > double-quoted-string >
 * interpolation > variable` occurs in practice — and the whole chain is kept so
 * `toCodeMirrorClass` can pick the innermost meaningful label.
 */
function flatten(nodes: readonly RootContent[], chain: string[] = []): TokenRun[] {
  const runs: TokenRun[] = []
  for (const node of nodes) {
    if (node.type === 'element') {
      const classes = (node.properties?.['className'] ?? []) as string[]
      runs.push(...flatten(node.children, [...chain, ...classes]))
    } else if (node.type === 'text') {
      runs.push({ text: node.value, chain })
    }
  }
  return runs
}

/**
 * Highlight `code`, returning the children for a fence's `<code>`.
 *
 * An unknown or absent language is an ordinary outcome, not an error — fences legitimately
 * carry no tag, and refractor ships a curated grammar set. Those render as plain text,
 * silently, which is what Typora does too.
 */
export function highlight(code: string, language: string | undefined): ElementContent[] {
  const resolved = resolveLanguage(language)
  if (!resolved) return [{ type: 'text', value: code }]

  let runs: TokenRun[]
  try {
    runs = flatten(refractor.highlight(code, resolved).children)
  } catch {
    // A grammar can throw on pathological input. Unhighlighted code is a fine outcome;
    // a failed build is not.
    return [{ type: 'text', value: code }]
  }

  const out: ElementContent[] = []
  for (const run of runs) {
    const cmClass = toCodeMirrorClass(run.chain)
    if (!cmClass) {
      // Deliberately unstyled — `punctuation` above all, which CodeMirror leaves in the
      // body colour. Wrapping it in a bare span would only add weight to every fence.
      out.push({ type: 'text', value: run.text })
      continue
    }
    // Prism's own names ride along too, so a Prism stylesheet would work here as well.
    const prismClasses = run.chain.filter((c) => c !== 'token')
    out.push({
      type: 'element',
      tagName: 'span',
      properties: { className: ['token', ...prismClasses, cmClass] },
      children: [{ type: 'text', value: run.text }],
    })
  }
  return out
}

export { resolveLanguage }
