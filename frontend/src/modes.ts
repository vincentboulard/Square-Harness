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
    mode: 'free', label: 'Default', glyph: '∀', kind: 'free',
    summary: 'Ask anything. The model picks the workflow and its effort and starts at once; cancel it if needed.',
    newLabel: 'New conversation', empty: 'No conversations yet.',
  },
  prove: {
    mode: 'prove', label: 'Prove', glyph: '⊢', kind: 'proof',
    summary: 'Solve the whole problem, review the written proof, repair a concrete objection.',
    newLabel: 'New proof', empty: 'No proofs yet in this workspace.',
  },
  // Not in the rail: the Default notebook answers with these, and older chats stay readable.
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
    mode: 'literature', label: 'Literature', glyph: '[1]', kind: 'research',
    summary: 'Bibliographical report with sources and the exact passages read.',
    newLabel: 'New literature report', empty: 'No literature reports yet.',
  },
  referee: {
    mode: 'referee', label: 'Review', glyph: '¶', kind: 'research',
    summary: 'Review a manuscript: a report with line-level citations.',
    newLabel: 'New review', empty: 'No reviews yet.',
  },
  writeup: {
    mode: 'writeup', label: 'Write-up', glyph: '§', kind: 'writeup',
    summary: 'Turn notes, drafts or PDFs into clean LaTeX in your own template style.',
    newLabel: 'New write-up', empty: 'No write-ups yet.',
  },
}
