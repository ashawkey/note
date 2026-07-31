import type { Parent, PhrasingContent } from 'mdast'

/*
 * Registers the nodes ./remark-typora-inline.ts invents.
 *
 * `==highlight==`, `X^2^` and `H~2~O` are Typora extensions, not CommonMark or GFM, so the
 * plugin produces node types mdast has never heard of. Without this declaration they are
 * untyped everywhere downstream — and concretely, mdast-util-to-hast keys its `Handlers`
 * map to *known* node types, so a handler for `highlight` is rejected outright and there is
 * no cast that makes the options object assignable.
 *
 * Augmenting the registry is the supported fix and the honest one: these really are mdast
 * nodes, they just come from an extension.
 */

declare module 'mdast' {
  interface Highlight extends Parent {
    type: 'highlight'
    children: PhrasingContent[]
  }

  interface Superscript extends Parent {
    type: 'superscript'
    children: PhrasingContent[]
  }

  interface Subscript extends Parent {
    type: 'subscript'
    children: PhrasingContent[]
  }

  // Both maps: phrasing content is where they occur, root content is what makes them
  // reachable from a generic tree walk.
  interface PhrasingContentMap {
    highlight: Highlight
    superscript: Superscript
    subscript: Subscript
  }

  interface RootContentMap {
    highlight: Highlight
    superscript: Superscript
    subscript: Subscript
  }
}
