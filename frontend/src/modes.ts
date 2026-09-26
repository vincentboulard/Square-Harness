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
// ∃ search for existence, [1] a citation, ¶ an edited manuscript, § a written section.
export const MODES: Record<Mode, ModeInfo> = {
  free: {
    mode: 'free', label: 'Free', glyph: '∀', kind: 'free',
    summary: 'Ask anything. The model suggests the right workflow; you confirm before it runs.',
    newLabel: 'New conversation', empty: 'No free conversations yet.',
  },
  prove: {
    mode: 'prove', label: 'Prove', glyph: '⊢', kind: 'proof',
    summary: 'Bounded proof search with a saved ledger of claims and objections.',
    newLabel: 'New proof', empty: 'No proofs yet in this workspace.',
  },
  critic: {
    mode: 'critic', label: 'Critic', glyph: '⊥', kind: 'chat',
    summary: 'Look for the first unjustified step and try counterexamples.',
    newLabel: 'New critique', empty: 'No critiques yet.',
  },
  explore: {
    mode: 'explore', label: 'Explore', glyph: '∃', kind: 'chat',
    summary: 'Explore approaches and connections, with heuristics labelled.',
    newLabel: 'New exploration', empty: 'No explorations yet.',
  },
  literature: {
    mode: 'literature', label: 'Literature', glyph: '[1]', kind: 'research',
    summary: 'Bibliographical report with sources and the exact passages read.',
    newLabel: 'New literature report', empty: 'No literature reports yet.',
  },
  referee: {
    mode: 'referee', label: 'Referee', glyph: '¶', kind: 'research',
    summary: 'Referee report on a pinned manuscript, with line-level citations.',
    newLabel: 'New referee report', empty: 'No referee reports yet.',
  },
  writeup: {
    mode: 'writeup', label: 'Write-up', glyph: '§', kind: 'writeup',
    summary: 'Turn notes, drafts or PDFs into clean LaTeX in your own template style.',
    newLabel: 'New write-up', empty: 'No write-ups yet.',
  },
}
