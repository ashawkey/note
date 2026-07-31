import GithubSlugger from 'github-slugger'
import type { Element, ElementContent, Root, RootContent } from 'hast'

import { highlight, resolveLanguage } from './highlight.ts'
import { renderMath, type MathFailure } from './math.ts'

/*
 * The DOM contract, applied.
 *
 * doc/dom-contract.md is the specification, and src/styles/ is what depends on it: those
 * stylesheets address the document the way Typora builds it, so every class below is load-
 * bearing. Emit this shape and kiwi.css and glassy.css style the notes correctly — and a
 * note renders here the same as it does in a Typora-family editor.
 *
 * An editor has a much harder time reaching this DOM, because it does not own the tree: it
 * has to push a live document model towards the target shape through whatever hooks the
 * editing framework exposes. A generator owns the tree outright and simply builds the shape,
 * which is why this is one walk — and why it emits no `md-focus`, no `.md-meta` markers and
 * no `.md-rawblock-input`, all of which exist only to make a document editable.
 */

const END_BLOCK = 'md-end-block'

/** Everything Typora puts on a code fence, flattened onto one <pre>. */
const FENCE_CLASS = [END_BLOCK, 'md-fences', 'ty-contain-cm', 'cm-s-inner', 'CodeMirror-wrap']

export interface Heading {
  depth: number
  text: string
  id: string
}

export interface TransformResult {
  headings: Heading[]
  mathFailures: MathFailure[]
  /** Language tags seen on fences, for reporting which ones had no grammar. */
  unknownLanguages: string[]
  /** Links to somewhere on the author's machine, which cannot be published. */
  unresolvableAssets: string[]
}

export interface TransformOptions {
  /**
   * Prepended to every relative URL in the note.
   *
   * A note becomes a *directory*: `notes/a/foo.md` is served at `/a/foo/`. Its images are
   * beside the Markdown file, at `notes/a/foo.assets/…`, and are copied to `dist/a/foo.assets/…`
   * — one level *above* the page. So every relative link needs a `../` to climb back out.
   * An `index.md` keeps its directory and needs none.
   */
  assetPrefix: string
}

/** URLs that are already absolute, or otherwise not ours to rewrite. */
const ABSOLUTE = /^(?:[a-z][a-z0-9+.-]*:|\/\/|\/|#|data:|mailto:)/i

/**
 * A path into the author's own filesystem — `C:\Users\…`, `file:///…`, a UNC share.
 *
 * Typora writes these when its image folder is set globally rather than per-document, and
 * the files are nowhere near the repository. They cannot be published, and silently emitting
 * a broken <img> hides that; the build reports them instead.
 *
 * The `%5C` alternatives are not paranoia: remark percent-encodes a backslash in a link
 * destination, so by the time the URL reaches here `C:\Users\…` reads `C:%5CUsers%5C…` and a
 * regex written against the literal path matches nothing at all.
 */
const LOCAL_ABSOLUTE = /^(?:[a-zA-Z]:(?:[\\/]|%5C|%2F)|\\\\|%5C%5C|file:)/i

function classesOf(node: Element): string[] {
  // Deliberately `unknown`: hast types className as a union wide enough that narrowing it
  // by `Array.isArray` first leaves the string branch unreachable.
  const raw: unknown = node.properties?.['className']
  if (Array.isArray(raw)) return raw.map(String)
  if (typeof raw === 'string') return raw.split(/\s+/).filter(Boolean)
  return []
}

function addClass(node: Element, ...names: string[]): void {
  node.properties ??= {}
  const existing = classesOf(node)
  for (const name of names) if (!existing.includes(name)) existing.push(name)
  node.properties['className'] = existing
}

/** Text content, ignoring comments, doctypes and the raw nodes that carry KaTeX's output. */
function textOf(node: RootContent | Root): string {
  if (node.type === 'text') return node.value
  if (node.type === 'element' || node.type === 'root') return node.children.map(textOf).join('')
  return ''
}

function isElement(node: RootContent | ElementContent, tagName?: string): node is Element {
  return node.type === 'element' && (tagName === undefined || node.tagName === tagName)
}

/**
 * Rewrite one hast tree in place into the DOM the themes expect.
 */
export function applyTyporaDom(tree: Root, options: TransformOptions): TransformResult {
  const slugger = new GithubSlugger()
  const headings: Heading[] = []
  const mathFailures: MathFailure[] = []
  const unknownLanguages = new Set<string>()
  const unresolvableAssets = new Set<string>()

  /** Re-point a relative URL at where the asset actually lands in dist/. */
  function rewriteUrl(node: Element, attribute: string): void {
    const value = node.properties?.[attribute]
    if (typeof value !== 'string' || !value) return

    if (LOCAL_ABSOLUTE.test(value)) {
      unresolvableAssets.add(value)
      return
    }
    if (ABSOLUTE.test(value)) return

    node.properties![attribute] = options.assetPrefix + value
  }

  /*
   * Footnote definitions are restructured rather than reclassed, so they are pulled out of
   * the tree here and appended as siblings at the end — see toFootnoteDefinitions.
   */
  let footnoteSection: Element | undefined

  function walk(node: RootContent | ElementContent, parent: Root | Element): void {
    if (!isElement(node)) return

    switch (node.tagName) {
      case 'p':
      case 'blockquote':
        addClass(node, END_BLOCK)
        break

      case 'h1':
      case 'h2':
      case 'h3':
      case 'h4':
      case 'h5':
      case 'h6': {
        // .md-heading is what several themes hang their own heading numbering off.
        addClass(node, END_BLOCK, 'md-heading')
        const text = textOf(node).trim()
        const id = slugger.slug(text || node.tagName)
        node.properties ??= {}
        node.properties['id'] = id
        headings.push({ depth: Number(node.tagName.slice(1)), text, id })
        break
      }

      case 'hr':
        addClass(node, END_BLOCK, 'md-hr')
        break

      case 'ul':
        addClass(node, 'ul-list')
        break

      case 'ol':
        addClass(node, 'ol-list')
        break

      case 'li':
        applyTaskItem(node)
        break

      case 'pre': {
        /*
         * remark-math spells display maths as a fenced block — `<pre><code
         * class="language-math math-display">` — so it has to be claimed before the fence
         * handler, which would otherwise syntax-highlight the TeX as if `math` were a
         * language and wrap it in `.md-fences`.
         */
        const fenced = node.children.find((child) => isElement(child, 'code'))
        if (fenced && isElement(fenced) && classesOf(fenced).includes('math-display')) {
          applyMath(node, textOf(fenced), true)
          return
        }
        if (applyFence(node)) return // children are already final
        break
      }

      case 'code':
        if (classesOf(node).includes('math-inline')) {
          applyMath(node, textOf(node), false)
          return
        }
        break

      case 'sup':
        // remark-rehype marks footnote references on the inner <a>; Typora puts the class
        // on the <sup> itself.
        if (node.children.some((c) => isElement(c, 'a') && 'dataFootnoteRef' in (c.properties ?? {})))
          addClass(node, 'md-footnote')
        break

      case 'img':
      case 'source':
      case 'video':
      case 'audio':
        rewriteUrl(node, 'src')
        break

      case 'a':
        // Only links to files travel with the note; `#section` and external URLs are already
        // correct, and ABSOLUTE screens both out.
        rewriteUrl(node, 'href')
        break

      case 'section':
        if (classesOf(node).includes('footnotes')) {
          footnoteSection = node
          // Removed from the flow here; the definitions are re-emitted at the end.
          const index = (parent.children as RootContent[]).indexOf(node as RootContent)
          if (index !== -1) (parent.children as RootContent[]).splice(index, 1)
          return
        }
        break

      default:
        break
    }

    for (const child of [...node.children]) walk(child, node)
  }

  /**
   * Fenced code.
   *
   * Typora renders fences through CodeMirror 5, so themes address the container as
   * `.md-fences` and the text as `.cm-s-inner .cm-keyword`. We emit one flattened <pre>
   * carrying all of those names at once rather than reproducing CodeMirror's scaffold.
   *
   * The <code> stays — unlike Typora, whose fences contain none — because it is what
   * scrolls. That is also why src/shell/bridge.css has to strip inline-code decoration back
   * off it: themes style `code` assuming it only ever appears inline.
   */
  function applyFence(pre: Element): boolean {
    const code = pre.children.find((child) => isElement(child, 'code'))
    if (!code || !isElement(code)) return false

    const languageClass = classesOf(code).find((name) => name.startsWith('language-'))
    const tag = languageClass?.slice('language-'.length)
    const source = textOf(code).replace(/\n$/, '')

    if (tag && !resolveLanguage(tag)) unknownLanguages.add(tag)

    pre.properties = { className: [...FENCE_CLASS] }
    // `lang` only when the fence is tagged: some themes key off
    // `[lang]:not([lang=""])::before` to draw their own label.
    if (tag) pre.properties['lang'] = tag

    code.properties = {}
    code.children = highlight(source, tag)

    // The language field. Typora puts an editable input here and positions it from app
    // code; a viewer has nothing to edit, so this is a plain span carrying the classes
    // themes style, placed by src/shell/compat.css.
    const children: ElementContent[] = [code]
    if (tag) {
      children.push({
        type: 'element',
        tagName: 'div',
        properties: { className: ['code-tooltip'] },
        children: [
          {
            type: 'element',
            tagName: 'span',
            properties: { className: ['ty-input', 'ty-input-after', 'ty-cm-lang-input'] },
            children: [{ type: 'text', value: tag }],
          },
        ],
      })
    }
    pre.children = children
    return true
  }

  /**
   * Task list items need *both* class names.
   *
   * base.css keys its layout on `.md-task-list-item` while some themes still target the
   * bare `.task-list-item`; emitting one alone visibly breaks the other half — the item
   * keeps its native bullet and the checkbox drops onto its own line. Likewise the checkbox
   * needs both the `checked` attribute and the property, because themes are split between
   * `[checked]` and `:checked`.
   */
  function applyTaskItem(li: Element): void {
    const box = li.children.find((child) => isElement(child, 'input'))
    if (!box || !isElement(box)) return
    if (box.properties?.['type'] !== 'checkbox') return

    const done = Boolean(box.properties['checked'])
    addClass(li, 'task-list-item', 'md-task-list-item', done ? 'task-list-done' : 'task-list-not-done')
    box.properties['disabled'] = true
    if (done) box.properties['checked'] = true
  }

  /**
   * Maths.
   *
   * The element remark-math produced is rewritten in place into the container Typora themes
   * position maths through, wrapping KaTeX's output.
   *
   * The display form omits `.md-rawblock-before/-input/-after`. Those hold the `$$` lines
   * and the TeX source, which an editor reveals when the caret enters the block. With
   * nothing to edit they would be three permanently hidden empty nodes per formula.
   */
  function applyMath(node: Element, tex: string, display: boolean): void {
    const rendered = renderMath(tex.trim(), display, mathFailures)

    if (display) {
      node.tagName = 'div'
      node.properties = { className: ['md-math-block', 'md-rawblock', END_BLOCK] }
      node.children = [
        {
          type: 'element',
          tagName: 'div',
          properties: { className: ['md-rawblock-container'] },
          children: [{ type: 'raw', value: rendered }],
        },
      ]
    } else {
      node.tagName = 'span'
      node.properties = { className: ['md-inline-math'] }
      node.children = [{ type: 'raw', value: rendered }]
    }
  }

  for (const child of [...tree.children]) walk(child, tree)

  if (footnoteSection) tree.children.push(...toFootnoteDefinitions(footnoteSection))

  return {
    headings,
    mathFailures,
    unknownLanguages: [...unknownLanguages].sort(),
    unresolvableAssets: [...unresolvableAssets],
  }
}

/**
 * remark-rehype's `<section class="footnotes"><ol><li>` becomes one
 * `div.footnotes.md-def-footnote` per note, which is Typora's shape: a flat sequence of
 * definitions, each carrying its own name, separator and content spans.
 */
function toFootnoteDefinitions(section: Element): Element[] {
  const list = section.children.find((child) => isElement(child, 'ol'))
  if (!list || !isElement(list)) return []

  const out: Element[] = []
  let index = 0

  for (const item of list.children) {
    if (!isElement(item, 'li')) continue
    index += 1

    // Drop the back-reference arrow: it is navigation for a page that has no other way
    // back, and the note's own superscript already links both ways.
    const content: ElementContent[] = []
    for (const child of item.children) {
      if (isElement(child, 'p')) {
        content.push(
          ...child.children.filter(
            (grand) => !(isElement(grand, 'a') && 'dataFootnoteBackref' in (grand.properties ?? {}))
          )
        )
      } else if (!isElement(child, 'a')) {
        content.push(child)
      }
    }

    // The definition is inline content in Typora, so the newlines remark-rehype leaves
    // around the list item's paragraph would otherwise show as gaps either side of the text.
    while (content[0]?.type === 'text' && !content[0].value.trim()) content.shift()
    while (content.at(-1)?.type === 'text' && !(content.at(-1) as { value: string }).value.trim())
      content.pop()
    const first = content[0]
    if (first?.type === 'text') first.value = first.value.replace(/^\s+/, '')
    const last = content.at(-1)
    if (last?.type === 'text') last.value = last.value.replace(/\s+$/, '')

    out.push({
      type: 'element',
      tagName: 'div',
      properties: {
        className: ['footnotes', 'md-def-footnote', 'md-end-block'],
        id: item.properties?.['id'],
      },
      children: [
        {
          type: 'element',
          tagName: 'span',
          properties: { className: ['md-def-name'] },
          children: [{ type: 'text', value: `[${index}]` }],
        },
        {
          type: 'element',
          tagName: 'span',
          properties: { className: ['md-def-split', 'md-def-f'] },
          children: [{ type: 'text', value: ' ' }],
        },
        {
          type: 'element',
          tagName: 'span',
          properties: { className: ['md-def-content'] },
          children: content,
        },
      ],
    })
  }

  return out
}
