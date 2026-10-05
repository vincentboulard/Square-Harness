import assert from 'node:assert/strict'
import test from 'node:test'
import { callPayload, modeQuery, updateCall, validCall, MODE_CALLS } from '../src/modeCalls.ts'

test('offers modes at the caret at the start, middle and end of a prompt', () => {
  for (const text of ['@pro', 'Please @pro show this', 'Show this (@pro)']) {
    const start = text.indexOf('@')
    assert.deepEqual(modeQuery(text, start + 4), { start, end: start + 4, query: 'pro' })
  }
  assert.equal(modeQuery('mail@prove.com', 10), null)
  assert.equal(modeQuery('@prove_extra', 6), null)
  assert.equal(modeQuery('@@prove', 7), null)
  assert.equal(modeQuery('@pro', 0, 4), null)
  assert.equal(MODE_CALLS.filter((item) => item.mode.startsWith('pro')).length, 1)
  assert.deepEqual(MODE_CALLS.map((item) => item.mode), ['prove', 'check', 'literature', 'referee'])
})

test('literal, unknown, and unconfirmed calls never produce execution metadata', () => {
  for (const text of ['@prove Show this', '@unknown Show this', 'Email a@prove.com']) {
    assert.equal(callPayload(text, null), undefined)
  }
  assert.equal(callPayload('@proven Show this', { mode: 'prove', start: 0, end: 6 }), undefined)
})

test('confirmed calls survive edits around them but lose confirmation when edited', () => {
  const text = 'Please @prove show this'
  const call = { mode: 'prove' as const, start: 7, end: 13 }
  assert.equal(validCall(text, call), true)
  assert.deepEqual(updateCall(text, 'Could you please @prove show this', call), { ...call, start: 17, end: 23 })
  assert.deepEqual(updateCall(text, 'Please @prove show that', call), call)
  assert.equal(updateCall(text, 'Please @critic show this', call), null)
  assert.equal(updateCall(text, 'Please @proven show this', call), null)
  assert.equal(updateCall(text, 'Please show this', call), null)
})

test('payload offsets count Unicode code points and target one selected occurrence', () => {
  const text = '🙂 @prove and @prove show this'
  const start = text.lastIndexOf('@prove')
  assert.deepEqual(callPayload(text, { mode: 'prove', start, end: start + 6 }), { mode: 'prove', start: 13 })
})
