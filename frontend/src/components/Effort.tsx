// Effort: one control instead of budget fields. Medium is the launch defaults;
// the other levels scale them, within the engine's own limits.
import { useEffect, useRef, type KeyboardEvent } from 'react'
import type { Defaults } from '../api'
import { count, plural } from '../format'

export type EffortKind = 'proof' | 'research' | 'writeup'
export type EffortLevel = 'low' | 'medium' | 'high' | 'xhigh' | 'brezis'

export const EFFORTS: { id: EffortLevel; label: string; factor: number; hint: string }[] = [
  { id: 'low', label: 'Low', factor: 0.35, hint: 'A quick look' },
  { id: 'medium', label: 'Medium', factor: 1, hint: 'The usual budget' },
  { id: 'high', label: 'High', factor: 2.5, hint: 'Longer, deeper work' },
  { id: 'xhigh', label: 'Extra high', factor: 6, hint: 'Hours of work' },
  { id: 'brezis', label: 'Brezis', factor: 20, hint: 'Everything the budget allows, in honour of Haïm Brezis' },
]

export type Limits = { rounds?: number; tokens: number; seconds: number; input_tokens?: number; requests?: number; chars?: number }

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value))
const thousands = (value: number) => Math.round(value / 1000) * 1000

export function effortLimits(kind: EffortKind, defaults: Defaults, level: EffortLevel): Limits {
  const f = EFFORTS.find((effort) => effort.id === level)!.factor
  if (kind === 'proof') {
    // Every attempt reserves a full solve and its review, so even Low keeps one of each.
    const cycle = defaults.proof_solve_tokens + defaults.proof_verify_tokens
    return {
      rounds: clamp(Math.round(defaults.proof_rounds * f), 1, 100),
      tokens: Math.max(Math.ceil(cycle / 1000) * 1000, thousands(defaults.proof_tokens * f)),
      seconds: Math.round(defaults.proof_seconds * f),
    }
  }
  const base: Limits = {
    tokens: Math.max(1024, thousands(defaults.research_tokens * f)),
    input_tokens: Math.max(2048, thousands(defaults.research_input_tokens * f)),
    seconds: Math.round(defaults.research_seconds * f),
  }
  if (kind === 'writeup') return base
  return {
    ...base,
    rounds: clamp(Math.round(defaults.research_rounds * f), 1, 100),
    requests: clamp(Math.round(defaults.research_requests * f), 0, 100),
    chars: clamp(thousands(defaults.research_chars * f), 1000, 1_000_000),
  }
}

export function sameLimits(a: Limits, b: Limits) {
  return (Object.keys(b) as (keyof Limits)[]).every((key) => a[key] === b[key])
}

function time(seconds: number) {
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`
  const hours = seconds / 3600
  return `${Number.isInteger(hours) ? hours : hours.toFixed(1)} h`
}

export function describeLimits(kind: EffortKind, limits: Limits) {
  if (kind === 'proof') return `Up to ${plural(limits.rounds!, 'attempt')}, ${count(limits.tokens)} tokens and ${time(limits.seconds)}`
  if (kind === 'writeup') return `Up to ${count(limits.tokens)} generated tokens and ${time(limits.seconds)}`
  return `Up to ${count(limits.rounds!)} rounds, ${count(limits.tokens)} generated tokens, ${count(limits.requests!)} web requests and ${time(limits.seconds)}`
}

function Sparkle({ className }: { className: string }) {
  return (
    <svg className={'sparkle ' + className} viewBox="0 0 16 16" aria-hidden="true">
      <path d="M8 0.8 9.6 6.4 15.2 8 9.6 9.6 8 15.2 6.4 9.6 0.8 8 6.4 6.4Z" />
    </svg>
  )
}

export function EffortSlider({ kind, defaults, level, onLevel, custom, compact }: {
  kind: EffortKind; defaults: Defaults; level: EffortLevel; onLevel: (level: EffortLevel) => void; custom?: boolean; compact?: boolean
}) {
  const index = EFFORTS.findIndex((effort) => effort.id === level)
  const refs = useRef<(HTMLButtonElement | null)[]>([])
  const focusNext = useRef(false)
  useEffect(() => {
    if (focusNext.current) { refs.current[index]?.focus(); focusNext.current = false }
  }, [index])
  const keys = (event: KeyboardEvent) => {
    const step = event.key === 'ArrowRight' || event.key === 'ArrowUp' ? 1 : event.key === 'ArrowLeft' || event.key === 'ArrowDown' ? -1 : 0
    if (!step) return
    event.preventDefault()
    focusNext.current = true
    onLevel(EFFORTS[clamp(index + step, 0, EFFORTS.length - 1)].id)
  }
  const brezis = level === 'brezis' && !custom
  return (
    <div className={'effort' + (compact ? ' effort-compact' : '') + (brezis ? ' effort-brezis' : '')}>
      <div className="effort-head">
        <span className="field-label">Effort</span>
        <span className="effort-now">
          {custom ? 'Custom limits' : EFFORTS[index].label}
          {brezis && <><Sparkle className="s1" /><Sparkle className="s2" /><Sparkle className="s3" /></>}
        </span>
      </div>
      <div className="effort-track" role="radiogroup" aria-label="Effort" onKeyDown={keys}>
        <span className="effort-line" aria-hidden="true"><span className="effort-fill" style={{ width: `${(index / (EFFORTS.length - 1)) * 100}%` }} /></span>
        {EFFORTS.map((effort, i) => (
          <button key={effort.id} ref={(node) => { refs.current[i] = node }} type="button" role="radio"
            aria-checked={effort.id === level} tabIndex={effort.id === level ? 0 : -1} title={effort.hint}
            className={'effort-stop' + (i <= index ? ' effort-stop-on' : '') + (effort.id === level ? ' effort-stop-current' : '') + (effort.id === 'brezis' ? ' effort-stop-brezis' : '')}
            onClick={() => onLevel(effort.id)}>
            <span className="effort-square" aria-hidden="true" />
            <span className="effort-label">{effort.label}{effort.id === 'brezis' && <Sparkle className="s0" />}</span>
          </button>
        ))}
      </div>
      {!compact && (
        <p className="field-hint effort-summary">
          {describeLimits(kind, effortLimits(kind, defaults, level))}{custom ? '; changed in Advanced limits below' : ''}.
          {brezis && ' The Brezis effort can run for hours; you can pause it at any time.'}
        </p>
      )}
    </div>
  )
}
