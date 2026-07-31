/*
 * Which language a note is indexed as.
 *
 * This exists for search, not for typography. Pagefind builds one index per language and
 * tokenises each with that language's rules — and the rules differ in a way that decides
 * whether a query can match at all. English is split on whitespace and punctuation, which is
 * correct for English and useless for Chinese: a run like `N皇后` has no spaces in it, so it
 * becomes a single token and a search for `皇后` matches nothing. Chinese needs a segmenter,
 * and Pagefind's Extended build has one — but only applies it to content marked as Chinese.
 *
 * So a note that is substantially Chinese must say so in `<html lang>`, or its text is
 * effectively unsearchable except by whole lines.
 *
 * A page can only be one language, so a mixed note is a compromise either way. The threshold
 * below picks the side that loses less: marking a Chinese note `en` makes its prose
 * unfindable, while marking an English note `zh` costs only stemming — `sorting` stops
 * matching `sort`, but every whole word still matches itself.
 */

const CJK = /[㐀-䶿一-鿿豈-﫿぀-ヿ]/gu

/**
 * Enough CJK characters to be worth segmenting.
 *
 * Absolute rather than proportional: a long English note with a paragraph of Chinese in it
 * still has a paragraph of Chinese that should be findable, and a proportional test would
 * bury it. Set high enough that a stray character — a name, a unit, an emoji-adjacent glyph
 * — does not reclassify an entire English note.
 */
const CJK_THRESHOLD = 8

/** BCP 47 primary subtag, which is all Pagefind reads. */
export type NoteLanguage = 'en' | 'zh'

export function detectLanguage(text: string): NoteLanguage {
  const matches = text.match(CJK)
  return matches && matches.length >= CJK_THRESHOLD ? 'zh' : 'en'
}
