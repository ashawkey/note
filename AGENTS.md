# AGENTS.md

## Project

- A static notes site: Markdown under `notes/` rendered into Typora's DOM at build time and
  styled by the Su and Qi liquid-glass palettes in `src/styles/`. No framework, no
  client-side routing.
- Node 20, npm with the committed `package-lock.json`. CI runs `npm ci && npm run build`.
- Everything renders at build time: formulas through KaTeX, code through refractor. The
  client bundle adds search, the drawer, the palette toggle and tree persistence — all
  enhancements over a page that is already complete without them. Keep it that way.

## Repository map

- `notes/`: the content, and the only directory a *reader-facing* change usually touches.
- `src/styles/`: the document stylesheets — `base.css` and `codemirror.css` underneath,
  `kiwi.css` for structure and type, `glassy.css` for the glass material, and the two
  palettes. `src/site/theme.ts` concatenates them into `dist/theme.css`.
- `src/render/`: Markdown to Typora DOM.
  - `typora-hast.ts` is the DOM-contract implementation and the file most likely to need
    changing. `pipeline.ts` assembles the unified processor.
  - `token-map.ts` translates Prism's token types to CodeMirror's; `languages.ts` resolves
    fence tags; `remark-typora-inline.ts` parses `==x==`, `X^2^`, `H~2~O`.
  - `typora-inline.d.ts`: registers that plugin's custom mdast nodes. Without it
    `mdast-util-to-hast` rejects handlers for them, since `Handlers` is keyed to known types.
- `src/site/`: `content.ts` walks `notes/` into a tree; `theme.ts` assembles the stylesheet.
- `src/shell/`: the page template, navigation tree, contents, the three CSS layers, and the
  client bundle. Layout decisions live here rather than in `src/styles/`.
- `scripts/`: `build.ts` (phase two), `dev.ts`, and the two verification harnesses.
  `serve.ts` is a shared in-process static server.
- `doc/dom-contract.md`: the spec `src/styles/` and `src/render/` both answer to. Read it
  before touching the renderer.
- `THIRD-PARTY.md`: what came from elsewhere and under which licence.

## Important contracts

- **Preserve the DOM contract.** `src/styles/` addresses the document the way Typora builds
  it, so the classes in `doc/dom-contract.md` are load-bearing. The failure mode is
  invisible: the page still renders, every selector still resolves, and the styling is
  silently gone. There is no automated check for it, so the rules are kept in sync by hand,
  and only on both sides at once: `src/styles/` and `src/render/typora-hast.ts`.
- **Nothing may sit between `#write` and a block.** `kiwi.css` uses `#write > *:first-child`.
- **Code tokens must carry `cm-*` classes**, not only Prism's. `kiwi.css` colours code
  through `.cm-s-inner .cm-keyword`; that is what `src/render/token-map.ts` is for.
- **Keep remark-gfm's `singleTilde: false`.** Typora reads `~x~` as subscript and reserves
  `~~x~~` for strikethrough. Turning it on breaks both.
- **A palette file is variables only.** `src/site/theme.ts` emits each palette's `:root`
  block three times under different `[data-theme]` scopes, and throws if a palette contains
  anything else. Structural rules belong in `kiwi.css` or `glassy.css`.
- **No `url()` in `src/styles/`.** Those files are concatenated into `dist/theme.css`, so a
  relative reference would resolve against the wrong directory. Asserted at build time.
  Anything needing its own base path is served from `public/` instead, as the webfont is.
- **Keep `src/shell/bridge.css` small.** Every rule in it is a place where the DOM and the
  stylesheets disagree and we patched it in CSS — debt, not design.
- **No `overflow` other than `visible` on `html` or `body`.** It makes them the scroll
  container and every `position: sticky` in the page silently sticks to that instead of the
  viewport, dropping the top bar and both side panels. This has already happened once.
- **No named inner functions inside `page.evaluate`.** tsx compiles with esbuild's
  `keepNames`, which wraps them in a `__name` helper that does not exist in the browser.
- **Resolve `import()` against `document.baseURI`.** A relative specifier inside the shell
  bundle resolves against the *module* — under `/assets/` — not the page, so
  `./pagefind/pagefind.js` asks for `/assets/pagefind/`. It fails only on some pages, because
  a note two levels deep has the prefix `../../`, which climbs out of `/assets/` and lands
  correctly by accident.
- **Keep Pagefind's `baseUrl: '/'`.** Left alone it infers a base from where its bundle
  loaded, so on a subpath deploy every result URL arrives already prefixed and `resolve()`
  prefixes it again. Exactly one place may know about deploy paths, and that is `resolve()`.
- **Test search through the UI, never through `page.evaluate`.** Both bugs above hid behind
  an API-level check, because `import()` inside an evaluated function resolves against the
  document while the same specifier in the bundle does not.
- **Highlight with the CSS Custom Highlight API, not `<mark>`.** Wrapping matches would
  insert elements inside `#write`, which the DOM contract forbids, split text nodes the
  stylesheets expect intact, and cut through KaTeX's positioned markup.
- **Page transitions stay declarative.** `@view-transition` plus a few `view-transition-name`
  values, no JavaScript. Do not reach for a router or click interception to "improve" it —
  the current version cannot break the no-script baseline, and any of those can.
  `@view-transition` is a top-level rule and must stay outside `@layer app`.
- **`::view-transition-group(*) { animation: none }` is what sets the duration.** Every
  pairing gets a `group` pseudo carrying the UA's default 250ms geometry animation, and it
  runs whether or not the geometry changed. Tuning only `old`/`new` does almost nothing:
  measured, cutting them 330ms → 195ms moved the transition 259ms → 250ms. Nothing here
  changes size between pages, so all the groups are off and the fade alone decides.
- **"Does not animate" means hiding the old half, not `animation: none` on both.** With no
  animation on either, both render at full opacity for the whole transition — two copies of a
  scrolled tree at different offsets, which is exactly the ghosting the arrangement avoids.
  Old gets `opacity: 0`, new gets `opacity: 1`.
- **Never name `.main` (or anything document-height).** A named element is snapshotted at its
  own size, so a full-height one scales a picture of an entire page against another, at
  mismatched scroll offsets — two documents superimposed, which reads as a failed redraw.
  Unnamed, it rides in the viewport-sized root snapshot, which is the right thing to fade.
- **`<html lang>` is a search setting, not just an accessibility one.** Pagefind tokenises per
  language: English splits on whitespace, so a Chinese run like `八皇后问题` becomes one token
  and `皇后` matches nothing. `src/site/language.ts` detects each note's language; do not
  hardcode `lang="en"` back into `src/shell/page.ts`.
- **Keep the `mergeIndex` call in `loadPagefind`.** Pagefind's runtime loads only the index
  matching the current page's language, so without it a search from an English note cannot
  see the Chinese ones. `data-langs` on `<html>` carries the full set from the build.
- **Never serve fonts with `no-store` from `scripts/serve.ts`.** It is the obvious dev-server
  default and it makes the browser re-download ~166 KB of subsets on every navigation, so
  local preview shows a flash of the fallback face that production does not have. Fonts,
  hashed bundles, KaTeX and the search index are immutable; HTML and `theme.css` revalidate.
- **Keep the ASCII font subsets preloaded, ahead of the stylesheet.** With ~194
  `unicode-range` subsets the browser cannot know what it needs until layout, so without the
  preload the first paint is always in the fallback face. `crossorigin` is required even
  same-origin — font fetches are CORS-mode, and without it the preload is simply discarded.
- **Non-Markdown files under `notes/` are copied to `dist/` verbatim**, and relative URLs in a
  note get a `../` because the note becomes a directory (`a/foo.md` → `/a/foo/`) while its
  images stay beside the Markdown (`a/foo.assets/`). An `index.md` keeps its directory and
  gets no prefix. Links into the author's filesystem are reported, not silently emitted —
  and note that remark percent-encodes backslashes, so `C:\…` arrives as `C:%5C…`.
- **Keep the background field named and static.** That is what lets the two pages fade out
  and in with no overlap: during a transition only the snapshots are painted, so without a
  pinned background the gap between them is an empty window rather than the page's own field.
- **No `scrollbar-gutter: stable` on `html`.** It reserves the scrollbar's width inside the
  layout and pulls all three docked panels off the edge of the screen. The small shift
  between scrolling and non-scrolling pages is the lesser problem. `overflow-y: scroll` is
  worse still — see the sticky rule above.
- **Honour `prefers-reduced-motion` in both directions.** Under `reduce` the view transition
  still runs — otherwise navigation flashes white — but every animation is `none`, and
  `scroll-behavior` must compute to `auto`.
- Assets are referenced by paths computed per page depth. Never write an absolute URL: the
  build has to keep working at a domain root, at a subpath, and from disk.

## Verification

Install with `npm ci`. There is no lint or unit-test script.

Choose the smallest relevant check:

- Types only: `npm run typecheck`.
- The full site: `npm run build` — the three phases: shell bundle, pages, search index.
- Anything visual: `npm run shoot`, then look at `test/shots/`. There is no automated DOM
  assertion anymore; the contract in `doc/dom-contract.md` is kept in sync by hand, and the
  screenshots are the closest thing to a check that it holds.
- Content or authoring changes: `npm run dev` and read the page.

The browser scripts launch the locally installed Chrome through Playwright
(`channel: 'chrome'`); they do not download a browser.
