import { refractor } from 'refractor/all'

/*
 * Language grammars for fenced code.
 *
 * `refractor/all` registers every Prism grammar (~270 languages), covering the tags these
 * notes actually use (`cmake`, `powershell`, `glsl`, `latex`, `nginx`, `protobuf`, …).
 * This runs at build time only, so the cost is a few milliseconds per fence and nothing
 * reaches the browser.
 *
 * If a language is missing, swap the import for `refractor/all`, or register a single
 * grammar here; the resolver below already treats "unknown" as a normal outcome. The build
 * reports every unrecognised fence tag it saw, so a missing grammar is visible rather than
 * silent.
 */

/**
 * Aliases people actually write in fences that Prism doesn't already know.
 *
 * Prism registers most short names itself (`js`, `py`, `rb`, `sh`, …), so this only fills
 * real gaps. Registering an alias for a grammar that isn't loaded would throw, hence the
 * guard.
 */
const EXTRA_ALIASES: Readonly<Record<string, readonly string[]>> = {
  markup: ['html', 'xml', 'svg'],
  bash: ['sh', 'shell', 'zsh', 'console'],
  javascript: ['js', 'mjs', 'cjs', 'node'],
  typescript: ['ts', 'mts', 'cts'],
  python: ['py'],
  ruby: ['rb'],
  csharp: ['cs', 'dotnet'],
  cpp: ['c++', 'cc', 'hpp'],
  yaml: ['yml'],
  markdown: ['md'],
  objectivec: ['objc'],
  makefile: ['make'],
  ini: ['toml', 'cfg', 'conf'],
}

let ready = false

function ensureReady(): void {
  if (ready) return
  ready = true
  for (const [language, aliases] of Object.entries(EXTRA_ALIASES)) {
    if (!refractor.registered(language)) continue
    for (const alias of aliases) {
      // Never clobber a name Prism already resolves — its own mapping is more accurate
      // than ours (`console` and `toml`, for instance, may gain real grammars later).
      if (!refractor.registered(alias)) refractor.alias(language, alias)
    }
  }
}

/**
 * Resolve a fence's language tag to a grammar refractor can highlight, or `undefined`.
 *
 * Unknown languages are an ordinary outcome, not an error: fences legitimately carry no
 * tag at all, or carry one for a grammar we don't ship. Typora renders those as plain
 * text, and so do we — silently. Milkdown's own Prism plugin logs a console warning per
 * unknown fence, which turns a normal document into a wall of noise.
 */
export function resolveLanguage(language: string | undefined | null): string | undefined {
  ensureReady()
  if (!language) return undefined
  const name = language.trim().toLowerCase()
  if (!name) return undefined
  return refractor.registered(name) ? name : undefined
}

export { refractor }
