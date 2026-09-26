// Status is a square: the mathematician's end-of-proof mark in its states.
import type { ClaimStatus } from '../api'

export type Variant =
  | 'reviewed' | 'gap' | 'refuted' | 'uncertain' | 'complete'
  | 'running' | 'paused' | 'spent' | 'error' | 'ready' | 'partial'

const BOX = { x: 2.75, y: 2.75, width: 10.5, height: 10.5, rx: 1 }

export function Square({ variant, size = 14, label }: { variant: Variant; size?: number; label?: string }) {
  return (
    <svg className={`sq sq-${variant}`} width={size} height={size} viewBox="0 0 16 16"
      role={label ? 'img' : undefined} aria-label={label} aria-hidden={label ? undefined : true}>
      {label && <title>{label}</title>}
      {variant === 'complete' ? (
        <rect {...BOX} className="sq-fill" />
      ) : (
        <rect {...BOX} className="sq-box" />
      )}
      {variant === 'reviewed' && <path d="M13.25 2.75V13.25H2.75Z" className="sq-fill" />}
      {variant === 'partial' && <rect x="2.75" y="8.5" width="10.5" height="4.75" className="sq-fill" />}
      {variant === 'refuted' && <path d="M5 5l6 6M11 5l-6 6" className="sq-mark" />}
      {variant === 'paused' && <path d="M6.5 5.5v5M9.5 5.5v5" className="sq-mark" />}
      {variant === 'spent' && <path d="M5.2 8h5.6" className="sq-mark" />}
      {variant === 'error' && <path d="M8 5v3.6M8 10.6v.4" className="sq-mark" />}
      {variant === 'running' && <rect x="5" y="5" width="6" height="6" className="sq-fill sq-pulse" />}
    </svg>
  )
}

type Look = { variant: Variant; label: string }

export function proofLook(status: string, running = false): Look {
  if (running) return { variant: 'running', label: 'Running' }
  switch (status) {
    case 'candidate_complete': return { variant: 'complete', label: 'Complete candidate, audited by the model' }
    case 'budget_exhausted': return { variant: 'spent', label: 'Budget used up' }
    case 'paused': return { variant: 'paused', label: 'Paused' }
    case 'running': return { variant: 'paused', label: 'Interrupted, can resume' }
    case 'ready': return { variant: 'ready', label: 'Ready to start' }
    case 'needs_context': return { variant: 'error', label: 'Needs a larger context' }
    case 'needs_recovery': return { variant: 'error', label: 'Needs recovery' }
    case 'stalled': return { variant: 'spent', label: 'Stalled' }
    case 'error': return { variant: 'error', label: 'Stopped by an error' }
    default: return { variant: 'uncertain', label: status }
  }
}

export function researchLook(status: string, running = false): Look {
  if (running) return { variant: 'running', label: 'Running' }
  switch (status) {
    case 'reviewed': return { variant: 'reviewed', label: 'Reviewed by the model' }
    case 'complete': return { variant: 'complete', label: 'Compiled and checked' }
    case 'partial': return { variant: 'partial', label: 'Partial report' }
    case 'budget_exhausted': return { variant: 'spent', label: 'Budget used up' }
    case 'paused': return { variant: 'paused', label: 'Paused' }
    case 'running': return { variant: 'paused', label: 'Interrupted, can resume' }
    case 'ready': return { variant: 'ready', label: 'Ready to start' }
    case 'error': return { variant: 'error', label: 'Stopped by an error' }
    default: return { variant: 'uncertain', label: status }
  }
}

export function claimLook(status: ClaimStatus | string): Look {
  switch (status) {
    case 'reviewed': return { variant: 'reviewed', label: 'Model-reviewed' }
    case 'gap': return { variant: 'gap', label: 'Gap' }
    case 'refuted': return { variant: 'refuted', label: 'Refuted' }
    default: return { variant: 'uncertain', label: 'Uncertain' }
  }
}

// Mirrors the engine: terminal jobs return their saved result instead of running.
export const proofResumable = (status: string) => ['ready', 'running', 'paused', 'error', 'interrupted', 'pending', 'active'].includes(status)
export const researchResumable = (status: string) => ['ready', 'running', 'paused', 'error'].includes(status)
