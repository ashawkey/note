# Notes

A static site for reading and sharing Markdown notes, rendered into Typora's DOM at build
time and styled by the Su and Qi liquid-glass palettes — so a note looks the same here as it
does in a Typora-family editor.

## Adding a note

Write a Markdown file under `notes/`, commit, push. That is the whole workflow; the GitHub
Actions workflow rebuilds and republishes the site on every push to `main`.

The folder structure under `notes/` *is* the structure of the site. There is no manifest, no
route table, no sidebar configuration, and no required frontmatter — `notes/algebra/groups.md`
becomes `/algebra/groups/` and appears in the sidebar under **Algebra**.

Everything else is inferred, with an override for when the inference is wrong:

| | Inferred from | Override |
| --- | --- | --- |
| Page title | first `#` heading, else the filename | `title:` in frontmatter |
| Sidebar order | numeric filename prefix (`01-`), else alphabetical | `order:` in frontmatter |
| Section name | that folder's `index.md` title, else the folder name | that file's `title:` |

A folder's `index.md` becomes its page; a folder without one gets a generated listing.
Names starting with `.` or `_` are skipped, so drafts and images can sit beside notes
without becoming pages. Add `hidden: true` to keep a note out of the sidebar while still
building and indexing it.

## What renders

GFM tables, footnotes, task lists, syntax-highlighted code, `$…$` and `$$…$$` formulas, and
Typora's inline extensions — `==highlight==`, `X^2^`, `H~2~O`.

Formulas and syntax highlighting are done **at build time**, so no JavaScript runs to
display a note, and the search index is built from real rendered text.

Deliberately not supported: Mermaid diagrams, raw HTML in Markdown, and wikilinks.

## Commands

```
npm run dev              build, serve on :4173, rebuild on change
npm run build            the full three-phase build into dist/
npm run serve            serve an existing dist/ on :4174, mounted at a subpath
npm run shoot            build, then screenshot both palettes into test/shots/
npm run typecheck        tsc --noEmit
```

The build is three phases, in order: `vite build` produces the shell bundle, `scripts/build.ts`
writes the pages, and `pagefind` indexes them. Running them out of order fails loudly —
`vite build` empties `dist/`.

## Deploying

Push to `main`. The workflow needs GitHub Pages set to "GitHub Actions" as its source, once,
under Settings → Pages.

There is no base-path setting to configure. Every URL the generator writes is relative to
the page it appears on, so the same `dist/` works at a domain root, at a project-page
subpath like `/note/`, and opened straight off disk. `npm run serve` mounts `dist/` at a
subpath specifically to keep that true.

## How it works

`src/styles/` is written against a particular DOM — Typora's — so that DOM is a
specification the renderer has to meet. It is written down in `doc/dom-contract.md`.
Breaking one is silent: the page still renders and every selector still resolves, they just
stop matching.

```
notes/                  your Markdown. The only directory you edit day to day.
site.config.ts          title, description, edit-link base
public/                 served verbatim (webfonts, favicon)
src/styles/             the document stylesheets, concatenated into dist/theme.css
src/render/             Markdown -> Typora DOM, at build time
src/site/               discovery, tree building, theme assembly
src/shell/              page template, nav tree, contents, CSS layers, client code
scripts/                build, dev, serve, and the screenshot harness
```

The cascade is four layers, in this order: `typora-base` (Typora's own base.css), then
`typora-theme` (kiwi + glassy), then `typora-bridge` (corrections for the one place our DOM
differs from Typora's), then `app` (this site's chrome). Declaring `app` last is what lets
`src/shell/shell.css` restyle anything without specificity games.

The chrome is drawn entirely from the theme's own tokens — `--lg-pane`, `--lg-sheen`,
`--lg-blur`, `--kiwi-ink`, `--kiwi-accent`. Not one colour is a literal, so the panels are
correct in both palettes because they ask the palette. The single light/dark distinction in
`shell.css` is how far the background field is dimmed, and even that reads the palette's own
`color-scheme` through `light-dark()` rather than restating when dark applies.

Two of this site's layout decisions override the document styles, and both live in
`@layer app` rather than being edited into `src/styles/`: the three panels are docked to the
edges of the screen rather than floating, and `#write` is stripped of `glassy.css`'s frosted
sheet so the note renders directly on the field. The blocks inside it — fences, tables,
quotes — keep their glass. Keeping the split means `glassy.css` goes on describing the
material while `shell.css` owns the layout.

## Search

<kbd>Ctrl</kbd>/<kbd>⌘</kbd>+<kbd>K</kbd> or <kbd>/</kbd> opens it; arrow keys and
<kbd>Enter</kbd> pick a result.

A result takes you to the line that matched, not just to the page: the link carries the
section anchor Pagefind found plus the query as `?q=`, and on arrival the term is located in
the text and scrolled to the middle of the window. Every occurrence stays highlighted for as
long as the term is in the search box — clear the box and the highlight goes with it, along
with the `?q=` in the address bar. Because the query lives in the URL, a link to a search hit
reproduces the highlight for whoever you send it to.

The highlight is painted through the CSS Custom Highlight API, so no elements are inserted
into the note — which is what keeps it from colliding with the DOM contract or with KaTeX's
markup. In a browser without that API the scroll still happens and only the colouring is
missing.

### Chinese, Japanese and Korean

Search indexes are language-specific, and the difference is not cosmetic. English is
tokenised on whitespace, which is useless for Chinese: a run like `八皇后问题` contains no
spaces, so under the English tokeniser it becomes a single token and searching `皇后` matches
nothing at all. Chinese needs a segmenter, which Pagefind's Extended build has — but applies
only to content marked as Chinese.

So the build detects each note's dominant language and writes it into `<html lang>`: a note
with enough CJK is indexed as `zh`, everything else as `en`. Because Pagefind's runtime loads
only the index matching the page you are standing on, the client then merges the other one,
and search covers the whole site from anywhere.

The threshold is absolute rather than proportional — a long English note with one Chinese
paragraph still has a paragraph that should be findable. A page can only be one language, so
a mixed note is a compromise either way; marking it `zh` costs English stemming (`sorting`
stops matching `sort`), while marking it `en` would make its Chinese unreachable.

## Motion

Moving between notes fades the document out and the next one in, while the top bar, the tree
and the background hold still. The contents pane is the exception, since its entries really
are different per note.

This is a cross-document view transition: one `@view-transition` rule and no JavaScript. No
router, no intercepted clicks, no fetch-and-swap, nothing to go wrong with scripting off. A
browser without support just navigates, exactly as before.

Two details in `shell.css` are what make it read as a fade rather than as a glitch, and both
are counter-intuitive enough to be worth knowing before editing them. The document is
deliberately **not** given a `view-transition-name`: naming it snapshots an element as tall
as the note, so the transition scales a picture of one whole page against another at
mismatched scroll offsets, and you see two documents superimposed. Left unnamed it rides in
the viewport-sized root snapshot instead. And the background field **is** named, purely so it
can be held still — that is what allows the two pages to fade out and in with no overlap,
because the gap between them shows the background instead of an empty window.

Jumping to a section — a contents entry, a footnote, any link to a `#heading` — scrolls
smoothly instead. Both respect `prefers-reduced-motion`: the transition still happens, so the
navigation does not flash white, but every part of it becomes instant.

The tree keeps its scroll position across navigations, stored per browser tab — otherwise
opening a note from the bottom of a long tree scrolls the tree away from the note you just
opened. When there is no stored position to restore, as on a shared link or a fresh tab, the
current note is scrolled into view instead.

## Fonts

The body face is LXGW WenKai, split into ~194 `unicode-range` subsets. That is what makes a
CJK family affordable — an English page fetches two or three files, not nine megabytes — but
it also means the browser cannot know which subsets it needs until the text is laid out, so
by default the first paint is always in a fallback face and swaps a moment later.

The build works out which subsets cover ASCII and emits `<link rel="preload">` for them,
ahead of the stylesheet that references them. Measured on a cold visit, the font is ready at
~15 ms against a first paint at ~320 ms, so text paints in the right face from the start. The
CJK subsets are deliberately left to load on demand: which ones a note needs depends on the
note.

On GitHub Pages the cache headers are GitHub's own (`max-age=600`), so within a session
navigation reuses the fonts and only revalidates occasionally. `npm run dev` and
`npm run serve` mirror that rather than sending `no-store`, which would otherwise make local
preview show a font flash on every page change that production never has.

## Themes

Two palettes, no picker: **Glassy Su** (light) and **Glassy Qi** (dark). Until you press the
toggle the site follows your system setting — no stored choice means no `[data-theme]`
attribute, and the stylesheet's `prefers-color-scheme` rule decides.

## Licences

Some of what is here was written elsewhere — Typora's base stylesheet, CodeMirror's, and
the LXGW WenKai webfont. `THIRD-PARTY.md` records what came from where and under which
licence.
