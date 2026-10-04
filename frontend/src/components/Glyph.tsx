import { MODES } from '../modes'
import type { Mode } from '../router'
import { SearchIcon } from './Icons'

// A literature check's mark in a Default conversation: a citation.
export const CHECK_GLYPH = '[1]'

/** A notebook's mark. Literature is drawn (a magnifying glass, sized by the font); the others are notation. */
export function Glyph({ mode }: { mode: Mode | 'check' }) {
  if (mode === 'literature') return <SearchIcon size="1em" />
  return <>{mode === 'check' ? CHECK_GLYPH : MODES[mode].glyph}</>
}
