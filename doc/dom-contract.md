# The DOM contract

`src/styles/` is written against a specific DOM, and `src/render/typora-hast.ts` emits it.
This file is the agreement between them.

It matters because breaking it is silent. Nothing throws, no type check fails, and the page
still renders — the selectors all still resolve, they just stop matching anything. Code
turns one flat colour, a task list keeps its bullets, a formula loses its placement.
There is no automated check for it; this file *is* the check, kept in sync by hand on both
sides.

## Where it comes from

The DOM is Typora's. The stylesheets began as Typora themes, so they address the document
the way Typora builds it: `#write` as the container, `md-*` classes on blocks, and — the
big one — CodeMirror 5's `cm-*` classes on syntax tokens.

That history is worth knowing for one practical reason: **the constraints are real but the
authority is now local**. If a rule here is inconvenient, it can be changed — but it has to
be changed on both sides, in `src/styles/` and in `src/render/typora-hast.ts`. The rules are
not arbitrary, they are just no longer external.

The practical benefit of keeping them is that a note renders identically here and in a
Typora-family editor, which is the point of the whole arrangement.

## Element contract

Everything below lives inside `<div id="write">`.

| Markdown | Required DOM | Why |
| --- | --- | --- |
| root | `<div id="write">`, the **direct** parent of block content | `kiwi.css` uses `#write > *:first-child`; an element in between breaks it |
| paragraph | `<p class="md-end-block">` | |
| heading | `<h1>`–`<h6>` with `md-end-block md-heading` and an `id` | the `id` anchors the contents pane |
| blockquote | `<blockquote class="md-end-block">` | |
| hr | `<hr class="md-end-block md-hr">` | |
| bullet / ordered list | `<ul class="ul-list">` / `<ol class="ol-list">` | |
| task item | `<li class="task-list-item md-task-list-item task-list-done\|task-list-not-done">` with a leading `<input type="checkbox">` | **both** class names — see below |
| code fence | `<pre class="md-end-block md-fences ty-contain-cm cm-s-inner CodeMirror-wrap" lang="…">` wrapping a `<code>` | `lang` only when the fence is tagged |
| fence tokens | `<span class="token … cm-keyword">` | CodeMirror's names, not only Prism's — see below |
| fence language label | `<div class="code-tooltip">` → `<span class="ty-cm-lang-input">` | positioned by `src/shell/compat.css` |
| table | native `<table><thead><tbody><th><td>` | |
| inline maths | `<span class="md-inline-math">` | |
| display maths | `<div class="md-math-block md-rawblock md-end-block">` → `.md-rawblock-container` | |
| footnote ref | `<sup class="md-footnote">` | |
| footnote def | `<div class="footnotes md-def-footnote md-end-block">` → `.md-def-name` + `.md-def-split.md-def-f` + `.md-def-content` | |
| `==x==` / `X^2^` / `H~2~O` | `<mark>` / `<sup>` / `<sub>` | plain tags |

## The three that are easy to get wrong

**Task lists need both class names.** `base.css` keys the layout on `.md-task-list-item`
while parts of `kiwi.css` target the bare `.task-list-item`. Emit one alone and the item
keeps its native bullet with the checkbox on its own line. The checkbox likewise needs both
the `checked` attribute and the property, since rules exist for `[checked]` and `:checked`.

**Fence tokens carry CodeMirror's class names.** Code is coloured through
`.cm-s-inner .cm-keyword`, `.cm-string` and friends — by far the largest styled surface in
`kiwi.css`, and the reason `src/render/token-map.ts` translates Prism's token types instead
of shipping Prism's own names. Prism's names ride along too, so a Prism stylesheet would
also work.

**`.md-fences` must not be a scroll container.** The `<code>` inside it is the scroller
(`src/shell/compat.css`), because the language label is absolutely positioned and would
otherwise scroll away with a long line.

## What a viewer deliberately omits

Typora's DOM carries several things that exist only so a document can be *edited*. None are
emitted here, and their absence is intentional:

- `.md-meta` / `.md-content` syntax markers, and the `.md-focus` reveal-on-focus behaviour.
- `.md-rawblock-before` / `-input` / `-after` — the `$$` lines and TeX source beside a
  rendered formula.
- An editable `<input>` for the fence language; a read-only `<span>` carries the classes.

## The one deviation

Typora's fences contain no `<code>` at all — the text lives in CodeMirror's own elements.
Ours do, because something has to scroll. That means themes' inline-`code` styling lands on
it, which `src/shell/bridge.css` strips back off. That file is the complete list of places
where the DOM and the stylesheets disagree, and it should stay short.
