import rehypeStringify from 'rehype-stringify'
import remarkFrontmatter from 'remark-frontmatter'
import remarkGemoji from 'remark-gemoji'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import { unified } from 'unified'
import { parse as parseYaml } from 'yaml'
import type { Parents, Root as MdastRoot } from 'mdast'
import type { Handler } from 'mdast-util-to-hast'
import type { Root } from 'hast'

import { applyTyporaDom, type Heading, type TransformOptions } from './typora-hast.ts'
import { remarkTyporaInline } from './remark-typora-inline.ts'
import type { MathFailure } from './math.ts'

export interface Frontmatter {
  title?: string
  order?: number
  /** Hide from the sidebar but still build and index the page. */
  hidden?: boolean
  [key: string]: unknown
}

export interface RenderedNote {
  html: string
  headings: Heading[]
  frontmatter: Frontmatter
  /** Resolved in priority order: frontmatter, then the first H1, then the filename. */
  title: string
  /** Plain text of the body, for excerpts. */
  text: string
  mathFailures: MathFailure[]
  unknownLanguages: string[]
  /** Links into the author's own filesystem, which cannot be published. */
  unresolvableAssets: string[]
}

/**
 * Frontmatter is read but never required.
 *
 * A note is a Markdown file and nothing else — no `title:` to remember, no schema to
 * satisfy. That is the whole point of requirement (5): adding a note means writing a file.
 * Everything the site needs it infers, and frontmatter only ever overrides an inference.
 */
function readFrontmatter(tree: MdastRoot): Frontmatter {
  const node = tree.children[0]
  if (node?.type !== 'yaml') return {}
  try {
    const parsed = parseYaml(node.value) as unknown
    return parsed && typeof parsed === 'object' ? (parsed as Frontmatter) : {}
  } catch {
    // A malformed block is the author's to fix, but it should not take the build down.
    return {}
  }
}

function firstHeadingText(tree: MdastRoot): string | undefined {
  for (const node of tree.children) {
    if (node.type === 'heading' && node.depth === 1) {
      const text = mdastText(node).trim()
      if (text) return text
    }
  }
  return undefined
}

function mdastText(node: unknown): string {
  const n = node as { value?: string; children?: unknown[] }
  if (typeof n.value === 'string') return n.value
  if (Array.isArray(n.children)) return n.children.map(mdastText).join('')
  return ''
}

/**
 * A mdast-to-hast handler mapping one of Typora's inline extensions onto a plain tag.
 *
 * The contract asks for exactly that here — `<mark>`, `<sup>`, `<sub>` — with no classes,
 * because Typora emits them as plain tags and themes style them as such.
 */
function inlineTag(tagName: 'mark' | 'sup' | 'sub'): Handler {
  return (state, node) => ({
    type: 'element',
    tagName,
    properties: {},
    children: state.all(node as Parents),
  })
}

function plainText(html: string): string {
  return html
    .replace(/<(script|style)[\s\S]*?<\/\1>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

const processor = unified()
  .use(remarkParse)
  .use(remarkFrontmatter, ['yaml'])
  .use(remarkGfm, {
    /*
     * Typora reads `~x~` as subscript (H~2~O) and reserves `~~x~~` for strikethrough, so
     * remark-gfm's default of accepting a single tilde would silently turn every subscript
     * into struck-through text.
     */
    singleTilde: false,
  })
  .use(remarkMath)
  .use(remarkGemoji)
  /*
   * Typora's inline extensions. Disabling single-tilde strikethrough above is what frees
   * `~` for subscript, so the order matters: this must come after remark-gfm.
   *
   * The plugin only parses, producing `highlight`/`superscript`/`subscript` mdast nodes.
   * Turning those into HTML is the handlers' job below, and the contract asks for plain
   * tags: <mark>, <sup>, <sub>.
   */
  .use(remarkTyporaInline)
  .use(remarkRehype, {
    // Our own raw nodes carry KaTeX's output; nothing from the Markdown source reaches
    // them. Raw HTML *in a note* is passed through as text by remark-rehype unless
    // rehype-raw is added, which it deliberately is not.
    allowDangerousHtml: true,
    footnoteLabel: 'Footnotes',
    clobberPrefix: '',
    handlers: {
      highlight: inlineTag('mark'),
      superscript: inlineTag('sup'),
      subscript: inlineTag('sub'),
    },
  })
  .use(rehypeStringify, { allowDangerousHtml: true })

/**
 * Render one note.
 *
 * `fallbackTitle` is the prettified filename, used only when the note gives no better
 * answer.
 */
export function renderNote(
  source: string,
  fallbackTitle: string,
  options: TransformOptions
): RenderedNote {
  const mdast = processor.parse(source) as MdastRoot
  const frontmatter = readFrontmatter(mdast)

  const hast = processor.runSync(mdast) as Root
  const { headings, mathFailures, unknownLanguages, unresolvableAssets } = applyTyporaDom(
    hast,
    options
  )
  const html = processor.stringify(hast)

  const title =
    (typeof frontmatter.title === 'string' && frontmatter.title.trim()) ||
    firstHeadingText(mdast) ||
    fallbackTitle

  return {
    html,
    headings,
    frontmatter,
    title,
    text: plainText(html),
    mathFailures,
    unknownLanguages,
    unresolvableAssets,
  }
}
