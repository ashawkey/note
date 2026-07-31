import type { Root, RootContent, PhrasingContent, Text } from 'mdast'
import type { Plugin } from 'unified'

/*
 * Typora's inline extensions: ==highlight==, X^2^ superscript, H~2~O subscript.
 *
 * None of these are CommonMark or GFM, and each would normally need a micromark syntax
 * extension — a substantial amount of code per marker. Instead this rewrites the parsed
 * tree: remark has already separated code spans, links, HTML and the like into their own
 * node types, so splitting plain `text` nodes on these delimiters cannot corrupt anything
 * that merely looks like one.
 *
 * `~` is available for subscript only because pipeline.ts disables remark-gfm's single-tilde
 * strikethrough. Turning that back on breaks both: `H~2~O` would become struck-through text.
 *
 * The node types this produces are registered in src/render/typora-inline.d.ts, and mapped
 * to <mark>/<sup>/<sub> by the handlers in pipeline.ts.
 */

interface Marker {
  /** mdast node type produced. */
  type: 'highlight' | 'superscript' | 'subscript'
  pattern: RegExp
}

const MARKERS: readonly Marker[] = [
  // Doubled delimiter: content may contain single `=` but not `==`.
  { type: 'highlight', pattern: /==(?!\s)((?:[^=]|=(?!=))+?)(?<!\s)==/g },
  // Single delimiters, and Typora treats whitespace as terminating them — `2^10` stays
  // literal, `X^2^` does not.
  { type: 'superscript', pattern: /\^(?!\s)([^\s^]+?)\^/g },
  { type: 'subscript', pattern: /~(?!\s)([^\s~]+?)~/g },
]

/** Split one text node on `marker`, leaving unmatched text as-is. */
function splitText(node: Text, marker: Marker): PhrasingContent[] {
  const out: PhrasingContent[] = []
  let last = 0
  marker.pattern.lastIndex = 0

  for (let m = marker.pattern.exec(node.value); m; m = marker.pattern.exec(node.value)) {
    const inner = m[1]
    if (inner === undefined) continue
    if (m.index > last) out.push({ type: 'text', value: node.value.slice(last, m.index) })
    out.push({ type: marker.type, children: [{ type: 'text', value: inner }] })
    last = m.index + m[0].length
  }

  if (out.length === 0) return [node]
  if (last < node.value.length) out.push({ type: 'text', value: node.value.slice(last) })
  return out
}

function transform(node: RootContent | Root): void {
  const parent = node as { children?: RootContent[] }
  if (!Array.isArray(parent.children)) return

  // Applied in order, each pass working on whatever text the previous one left behind, so
  // `==H~2~O==` nests correctly.
  for (const marker of MARKERS) {
    const next: RootContent[] = []
    for (const child of parent.children) {
      if (child.type === 'text') next.push(...(splitText(child, marker) as RootContent[]))
      else next.push(child)
    }
    parent.children = next
  }

  for (const child of parent.children) transform(child)
}

export const remarkTyporaInline: Plugin<[], Root> = function remarkTyporaInline() {
  return (tree: Root) => {
    transform(tree)
  }
}
