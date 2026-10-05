import type { Mode } from './router'

export type ModeInfo = {
  mode: Mode
  label: string
  glyph: string
  kind: 'free' | 'proof' | 'chat' | 'research' | 'writeup'
  summary: string
  newLabel: string
  empty: string
}

// Glyphs are the subject's own notation: ∀ any task, ⊢ proves, ⊥ contradiction,
// ∃ search for existence, a magnifying glass for the literature (drawn: components/Glyph.tsx),
// ¶ an edited manuscript, § a written section. A literature check in Default is marked [1], a citation.
export const MODES: Record<Mode, ModeInfo> = {
  free: {
    mode: 'free', label: 'Assistant', glyph: '∀', kind: 'free',
    summary: 'Discuss mathematics, get a quick answer, or work through a harder task with focused help.',
    newLabel: 'New conversation', empty: 'No conversations yet.',
  },
  prove: {
    mode: 'prove', label: 'Prove', glyph: '⊢', kind: 'proof',
    summary: 'Solve the whole problem, review the written proof, repair a concrete objection.',
    newLabel: 'New proof', empty: 'No proofs yet in this workspace.',
  },
  // Not in the rail: the Assistant can use these, and older chats stay readable.
  critic: {
    mode: 'critic', label: 'Critique', glyph: '⊥', kind: 'chat',
    summary: 'Look for the first unjustified step and try counterexamples.',
    newLabel: 'New conversation', empty: 'No conversations yet.',
  },
  explore: {
    mode: 'explore', label: 'Exploration', glyph: '∃', kind: 'chat',
    summary: 'Explore approaches and connections, with heuristics labelled.',
    newLabel: 'New conversation', empty: 'No conversations yet.',
  },
  literature: {
    mode: 'literature', label: 'Literature', glyph: '⌕', kind: 'research',
    summary: 'A verified reading list: entry points, then the core literature by theme.',
    newLabel: 'New reading list', empty: 'No reading lists yet.',
  },
  referee: {
    mode: 'referee', label: 'Review', glyph: '¶', kind: 'research',
    summary: 'Check a proof, referee a manuscript or explain a result, with line-level citations.',
    newLabel: 'New review', empty: 'No reviews yet.',
  },
  writeup: {
    mode: 'writeup', label: 'Write-up', glyph: '§', kind: 'writeup',
    summary: 'Turn notes, drafts or PDFs into clean LaTeX in your own template style.',
    newLabel: 'New write-up', empty: 'No write-ups yet.',
  },
}
