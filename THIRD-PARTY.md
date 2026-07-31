# Third-party material

Not everything in this repository was written for it. This file records what came from
elsewhere, so the provenance survives even though the files are now edited freely alongside
the rest of the codebase.

## `src/styles/base.css`

Typora's base stylesheet, which the document styles are written against — it supplies
`--bg-color`, `--text-color`, `--select-text-bg-color` and the `.md-*` layout primitives
that `kiwi.css` builds on. Retained because dropping it would mean reimplementing those
primitives, and the DOM would still be Typora's either way.

## `src/styles/codemirror.css`

CodeMirror 5's stylesheet, MIT-licensed. Present because fenced code carries CodeMirror's
`cm-*` token classes — see `doc/dom-contract.md` — and this provides the defaults that
`kiwi.css`'s palette overrides.

## `public/fonts/lxgw-wenkai/`

LXGW WenKai (霞鹜文楷) by LXGW, under the SIL Open Font License 1.1. The licence travels
with the files: `public/fonts/lxgw-wenkai/LICENSE` and `OFL.txt`. Subset by `unicode-range`,
so a page downloads only the ranges it actually sets — which is why 9 MB on disk is a
handful of small requests in practice.

Redistribution as a webfont is permitted by the OFL. If the family is ever swapped out,
remove these files with it.

## `src/styles/kiwi.css`, `glassy.css`, `glassy-su.css`, `glassy-qi.css`

Originally written for [kiwidown](https://github.com/ashawkey/kiwidown) as Typora themes,
by the same author as this site. They have since diverged and are maintained here.

## Runtime dependencies

Everything under `node_modules/` keeps its own licence; `package.json` is the list. The two
that ship bytes to the reader rather than only running at build time are **KaTeX** (MIT —
`dist/katex/` carries its stylesheet and glyph fonts) and **Pagefind** (MIT — `dist/pagefind/`
carries its index and runtime).
