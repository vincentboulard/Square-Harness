// Run with: npm test
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { describeLimits, EFFORTS, effortLimits, isLevel } from '../src/effort.ts'
import type { Defaults } from '../src/api.ts'

const defaults: Defaults = {
  ctx: 40960, predict: 4096, think: true,
  proof_rounds: 3, proof_tokens: 120000, proof_seconds: 1800,
  proof_solve_tokens: 32768, proof_verify_tokens: 16384, proof_repair_tokens: null, proof_min_solve_tokens: null,
  research_rounds: 6, research_tokens: 24000, research_input_tokens: 100000,
  research_seconds: 900, research_requests: 12, research_chars: 30000,
}

test('the levels are the agreed budgets', () => {
  assert.deepEqual(EFFORTS.map((e) => e.id), ['low', 'medium', 'high', 'xhigh', 'brezis'])
  const low = effortLimits('proof', defaults, 'low')
  assert.equal(low.rounds, 1)
  assert.equal(low.seconds, 60)
  assert.equal(low.tokens, 30000)
  assert.equal(effortLimits('proof', defaults, 'medium').rounds, 3)
  const brezis = effortLimits('research', defaults, 'brezis')
  assert.deepEqual([brezis.rounds, brezis.seconds, brezis.tokens], [10, 7200, 200000])
})

test('budgets grow with the level', () => {
  for (const kind of ['proof', 'research', 'writeup'] as const) {
    const levels = EFFORTS.map((e) => effortLimits(kind, defaults, e.id))
    for (let i = 1; i < levels.length; i++) {
      assert.ok(levels[i].tokens > levels[i - 1].tokens, kind + ' tokens')
      assert.ok(levels[i].seconds > levels[i - 1].seconds, kind + ' time')
    }
  }
})

test('a small proof budget still holds one full solve and its review', () => {
  const low = effortLimits('proof', defaults, 'low')
  assert.ok(low.solve_tokens! + low.verify_tokens! <= low.tokens)
  assert.ok(low.solve_tokens! < defaults.proof_solve_tokens)
  const medium = effortLimits('proof', defaults, 'medium')
  assert.deepEqual([medium.solve_tokens, medium.verify_tokens], [32768, 16384])
})

test('launch repair floors are respected, raising the budget if needed', () => {
  const floors = { ...defaults, proof_repair_tokens: 28000 }
  const low = effortLimits('proof', floors, 'low')
  assert.ok(low.solve_tokens! >= 28000)
  assert.ok(low.solve_tokens! + low.verify_tokens! <= low.tokens)
})

test('write-ups have no rounds; reports have web limits', () => {
  assert.equal(effortLimits('writeup', defaults, 'high').rounds, undefined)
  const report = effortLimits('research', defaults, 'high')
  assert.ok(report.requests! > 0 && report.chars! > 0 && report.input_tokens! > report.tokens)
})

test('summaries and level names', () => {
  assert.match(describeLimits('proof', effortLimits('proof', defaults, 'low')), /^Up to 1 attempt, 30.000 tokens and 1 min$/)
  assert.match(describeLimits('research', effortLimits('research', defaults, 'brezis')), /10 rounds.*2 h$/)
  assert.ok(isLevel('brezis'))
  assert.ok(!isLevel('extreme'))
})
