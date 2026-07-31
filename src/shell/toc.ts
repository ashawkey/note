import type { Heading } from '../render/typora-hast.ts'

/*
 * The in-page contents, in Typora's `[TOC]` markup.
 *
 * `.md-toc` is styled by nearly every Typora theme, and kiwi.css and glassy.css are no
 * exception — kiwi.css:442-468 sets its type and spacing, glassy.css:244 makes it a glass
 * pane alongside fences and tables. Emitting Typora's own structure means the contents
 * needs no styling of its own and stays right in both palettes for free.
 */

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

/**
 * Render `headings` as Typora's table of contents.
 *
 * H1 is skipped: it is the note's title, shown in the page header already, so listing it
 * would make every contents start with a restatement of where you are.
 */
export function renderToc(headings: readonly Heading[]): string {
  const listed = headings.filter((heading) => heading.depth >= 2 && heading.depth <= 4)
  if (listed.length < 2) return '' // one entry is a label, not a contents

  const items = listed
    .map(
      (heading) =>
        `<span class="md-toc-item md-toc-h${heading.depth}" data-ref="${escapeHtml(heading.id)}">` +
        `<a class="md-toc-inner" href="#${escapeHtml(heading.id)}">${escapeHtml(heading.text)}</a>` +
        `</span>`
    )
    .join('')

  return `<div class="md-toc"><p class="md-toc-content">${items}</p></div>`
}
