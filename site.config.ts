/*
 * Site-wide settings.
 *
 * Deliberately tiny. There is no sidebar configuration, no route table and no navigation
 * manifest here, because the notes/ directory already is all three — the structure on disk
 * is the structure of the site.
 *
 * Note what is *not* here: a base path. Every URL the generator writes is relative to the
 * page it is written on, so the same dist/ works at a domain root, at a project-page
 * subpath like /note/, and opened straight off disk. Renaming the repository changes
 * nothing.
 */
export const siteConfig = {
  /** Shown in the top bar and appended to every page title. */
  title: 'Notes',

  /** Fallback <meta name="description"> for pages with no prose to excerpt. */
  description: 'Personal notes.',

  /**
   * Base URL for the "Edit this page" link, or undefined to omit it.
   * e.g. 'https://github.com/ashawkey/note/edit/main'
   */
  editBaseUrl: undefined as string | undefined,
}
