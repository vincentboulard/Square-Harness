import type { ReviewVariant } from './api'
import type { EffortLevel } from './effort'
import type { Variant } from './components/Square'

/** The three kinds of review behind the Review notebook; the engine kind stays 'referee'. */
// 'journal' is the detailed review: the review plus a check of the proofs.
export const VARIANTS: ReviewVariant[] = ['quick', 'review', 'journal', 'explain']

export const VARIANT_NAMES: Record<ReviewVariant, string> = {
  quick: 'Quick check', review: 'Review', journal: 'Detailed review', explain: 'Explanation',
}

/** A quick check ends with a verdict on the proof, shown with the notebook's squares. */
export const VERDICTS: Record<string, { variant: Variant; label: string }> = {
  no_issue_found: { variant: 'reviewed', label: 'No issue found' },
  issues_found: { variant: 'refuted', label: 'Issues found' },
  issues_alleged: { variant: 'gap', label: 'Issues alleged, not re-checked' },
  uncertain: { variant: 'uncertain', label: 'Uncertain' },
  unavailable: { variant: 'ready', label: 'No usable check' },
}

/** Workflow steps of each kind, as the engine names its phases. */
export const VARIANT_STEPS: Record<ReviewVariant, string[]> = {
  quick: ['plan', 'extract', 'verify', 'confirm', 'done'],
  review: ['plan', 'read', 'scope', 'literature', 'write', 'done'],
  journal: ['plan', 'read', 'scope', 'check', 'confirm', 'literature', 'write', 'done'],
  explain: ['plan', 'extract', 'explain', 'fill', 'check', 'fix', 'done'],
}

export const VARIANT_STEP_NAMES: Record<string, string> = {
  plan: 'Map', extract: 'Find', verify: 'Check', confirm: 'Re-check', read: 'Read', scope: 'Overview', check: 'Proofs',
  literature: 'Literature', presentation: 'Presentation', write: 'Write', explain: 'Explain', fill: 'Fill', fix: 'Fix', done: 'Done',
}

/** The effort each kind starts from: a quick check wants two passes, a referee report depth. */
export const VARIANT_EFFORT: Record<ReviewVariant, EffortLevel> = { quick: 'medium', review: 'medium', journal: 'high', explain: 'low' }

/** A detailed review also checks the proofs: twice the tokens, context and time of a review (gui/effort.py). */
export const VARIANT_SCALE: Record<ReviewVariant, number> = { quick: 1, review: 1, journal: 2, explain: 1 }

export const VARIANT_CAVEATS: Record<ReviewVariant, string> = {
  quick: 'Model check, not a certificate. Each objection links to the proof lines it concerns; the proof was not rewritten.',
  review: 'Model draft of a referee report on understanding and presentation; the proofs were not checked. Links open the manuscript lines.',
  journal: 'Model draft of a detailed referee report. The comments on proofs come from checked findings; links open the manuscript lines they cite.',
  explain: 'Model explanation. Links open the source lines each step explains; an accuracy pass checks it against the source when the effort allows.',
}
