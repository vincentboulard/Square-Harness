// Effort levels: one control instead of budget fields (no React; testable with node).
import type { Defaults } from './api'

export type EffortKind = 'proof' | 'research' | 'writeup'
export type EffortLevel = 'low' | 'medium' | 'high' | 'xhigh' | 'brezis'

// The same budgets for every kind of job. Tries are proof attempts or investigation
// rounds; tokens are generated tokens, thinking included. Reports also get input
// tokens (every call's context counts again), web requests and evidence characters.
export const EFFORTS: {
  id: EffortLevel; label: string; hint: string
  tries: number; minutes: number; tokens: number; input: number; requests: number; chars: number
}[] = [
  { id: 'low', label: 'Low', hint: 'One quick try', tries: 1, minutes: 1, tokens: 30_000, input: 120_000, requests: 4, chars: 15_000 },
  { id: 'medium', label: 'Medium', hint: 'Three tries, the usual choice', tries: 3, minutes: 15, tokens: 60_000, input: 240_000, requests: 12, chars: 30_000 },
  { id: 'high', label: 'High', hint: 'Five tries for harder work', tries: 5, minutes: 30, tokens: 100_000, input: 400_000, requests: 24, chars: 60_000 },
  { id: 'xhigh', label: 'Extra high', hint: 'Seven tries, up to an hour', tries: 7, minutes: 60, tokens: 150_000, input: 600_000, requests: 40, chars: 100_000 },
  { id: 'brezis', label: 'Brezis', hint: 'Ten tries, up to two hours, in honour of Haïm Brezis', tries: 10, minutes: 120, tokens: 200_000, input: 800_000, requests: 60, chars: 150_000 },
]

export const LEVELS = EFFORTS.map((effort) => effort.id)

export type Limits = {
  rounds?: number; tokens: number; seconds: number; input_tokens?: number; requests?: number; chars?: number
  solve_tokens?: number; verify_tokens?: number
}

const count = (value: number) => new Intl.NumberFormat(undefined).format(Math.round(value))

export function isLevel(value: unknown): value is EffortLevel {
  return typeof value === 'string' && (LEVELS as string[]).includes(value)
}

/** A proof needs room for one full solve and its review: fit the per-call ceilings in the budget. */
function proofCeilings(defaults: Defaults, tokens: number) {
  const solve = defaults.proof_solve_tokens
  const verify = defaults.proof_verify_tokens
  if (tokens >= solve + verify) return { tokens, solve_tokens: solve, verify_tokens: verify }
  // Launch floors for repairs stay valid: the solve ceiling never goes below them.
  const floor = Math.max(128, defaults.proof_repair_tokens ?? 0, defaults.proof_min_solve_tokens ?? 0)
  const fitted = Math.min(solve, Math.max(floor, Math.floor((tokens * solve) / (solve + verify))))
  const review = Math.min(verify, Math.max(128, tokens - fitted))
  return { tokens: Math.max(tokens, fitted + review), solve_tokens: fitted, verify_tokens: review }
}

export function effortLimits(kind: EffortKind, defaults: Defaults, level: EffortLevel): Limits {
  const effort = EFFORTS.find((item) => item.id === level) || EFFORTS[1]
  const seconds = effort.minutes * 60
  if (kind === 'proof') return { rounds: effort.tries, seconds, ...proofCeilings(defaults, effort.tokens) }
  const base: Limits = { tokens: effort.tokens, input_tokens: effort.input, seconds }
  if (kind === 'writeup') return base
  return { ...base, rounds: effort.tries, requests: effort.requests, chars: effort.chars }
}

export function minutesLabel(seconds: number) {
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`
  const hours = seconds / 3600
  return `${Number.isInteger(hours) ? hours : hours.toFixed(1)} h`
}

export function describeLimits(kind: EffortKind, limits: Limits) {
  const tries = (n: number, one: string) => `${count(n)} ${n === 1 ? one : one + 's'}`
  if (kind === 'proof') return `Up to ${tries(limits.rounds!, 'attempt')}, ${count(limits.tokens)} tokens and ${minutesLabel(limits.seconds)}`
  if (kind === 'writeup') return `Up to ${count(limits.tokens)} generated tokens and ${minutesLabel(limits.seconds)}`
  return `Up to ${tries(limits.rounds!, 'round')}, ${count(limits.tokens)} generated tokens, ${count(limits.requests!)} web requests and ${minutesLabel(limits.seconds)}`
}
