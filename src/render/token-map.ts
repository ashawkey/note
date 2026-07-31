/*
 * Prism token types → CodeMirror 5 token classes.
 *
 * Code is coloured the way Typora colours it — through CodeMirror 5's vocabulary,
 * `.cm-s-inner .cm-keyword`, `.cm-string` and so on. That is the largest styled surface in
 * src/styles/kiwi.css. Emitting Prism's own `token keyword` names would leave all of it
 * unmatched and code would render in one flat colour.
 *
 * So the highlighter tokenises with Prism and relabels the output with CodeMirror's
 * vocabulary. Both sets of class names are emitted, so a Prism stylesheet would also work
 * if anyone wants one.
 *
 * The mapping targets are what src/styles/kiwi.css actually colours; the sources are the
 * token chains Prism really produces, which are undocumented and differ per grammar. If you
 * add a grammar and its code comes out flat, this map is the place to look — dump a chain
 * from src/render/highlight.ts and add whatever it turns out to be called.
 */

/**
 * The CodeMirror 5 token classes worth mapping onto.
 *
 * The first group is what kiwi.css colours; the ones after it are progressively rarer, down
 * to the diff and Markdown classes. Mapping onto a class nothing styles is harmless but
 * pointless, so grep src/styles/ before adding one.
 */
type CodeMirrorToken =
  | 'keyword'
  | 'atom'
  | 'number'
  | 'def'
  | 'variable'
  | 'variable-2'
  | 'variable-3'
  | 'type'
  | 'property'
  | 'operator'
  | 'comment'
  | 'string'
  | 'string-2'
  | 'meta'
  | 'qualifier'
  | 'builtin'
  | 'tag'
  | 'attribute'
  | 'link'
  | 'header'
  | 'quote'
  | 'positive'
  | 'negative'
  | 'strong'
  | 'em'

/**
 * Prism token type → CodeMirror class.
 *
 * Anything absent is deliberately unstyled. `punctuation` is the big one: it is the single
 * most common token Prism emits, and CodeMirror leaves it in the body colour. Mapping it
 * would tint every brace and semicolon in every fence.
 */
const TOKEN_MAP: Readonly<Record<string, CodeMirrorToken>> = {
  keyword: 'keyword',
  comment: 'comment',

  // Strings, and the pieces Prism splits them into.
  string: 'string',
  'double-quoted-string': 'string',
  'single-quoted-string': 'string',
  'triple-quoted-string': 'string',
  char: 'string',
  'attr-value': 'string',

  // CodeMirror puts regexes in string-2, distinct from ordinary strings.
  regex: 'string-2',
  'regex-source': 'string-2',
  'regex-delimiter': 'string-2',
  'regex-flags': 'string-2',
  'regex-literal': 'string-2',

  number: 'number',

  // Literals and interned values all land on atom, as they do in CodeMirror.
  boolean: 'atom',
  constant: 'atom',
  null: 'atom',
  symbol: 'atom',
  entity: 'atom',
  'named-entity': 'atom',

  // Definitions: names being introduced rather than referenced.
  function: 'def',
  'function-definition': 'def',
  'method-definition': 'def',
  atrule: 'def',
  rule: 'def',

  // Types. CodeMirror's own modes use variable-3 for this far more widely than `type`, and
  // kiwi.css follows suit — it colours variable-3 and never colours type at all.
  'class-name': 'variable-3',
  'class-name-definition': 'variable-3',
  'type-definition': 'variable-3',
  'type-hint': 'variable-3',
  'return-type': 'variable-3',

  builtin: 'builtin',
  variable: 'variable',

  // Locals and qualified names — CodeMirror's variable-2.
  parameter: 'variable-2',
  namespace: 'variable-2',
  package: 'variable-2',

  property: 'property',
  'literal-property': 'property',
  key: 'property',

  operator: 'operator',
  'attr-equals': 'operator',

  tag: 'tag',
  'doctype-tag': 'tag',
  'attr-name': 'attribute',
  attribute: 'attribute',
  selector: 'qualifier',

  // Everything out-of-band: preprocessor lines, annotations, shebangs, diff hunk headers.
  important: 'meta',
  doctype: 'meta',
  prolog: 'meta',
  directive: 'meta',
  'directive-hash': 'meta',
  macro: 'meta',
  shebang: 'meta',
  annotation: 'meta',
  decorator: 'meta',
  coord: 'meta',
  cdata: 'comment',

  url: 'link',

  // diff
  inserted: 'positive',
  deleted: 'negative',

  // markdown
  title: 'header',
  blockquote: 'quote',
  bold: 'strong',
  italic: 'em',
}

/**
 * Pick the CodeMirror class for one token, given its full nesting chain.
 *
 * Prism nests tokens, so a run of text arrives labelled with every class of every ancestor
 * — `php > language-php > string > double-quoted-string > interpolation > variable` is a
 * real chain from the probe. The innermost meaningful label is the right one, so the chain
 * is read from the inside out and the first recognised type wins.
 *
 * This is also what makes the unmapped entries work: structural wrappers like
 * `language-php`, `regex-source` or `string-literal` simply don't match, and the search
 * continues outward until something does.
 */
export function toCodeMirrorClass(chain: readonly string[]): string | undefined {
  for (let i = chain.length - 1; i >= 0; i--) {
    const token = TOKEN_MAP[chain[i] as string]
    if (token) return `cm-${token}`
  }
  return undefined
}
