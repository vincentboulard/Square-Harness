import { test } from 'node:test'
import assert from 'node:assert/strict'
import { api, type TaskSummary, type TranscriptItem } from '../src/api.ts'
import { assistantResumable, routeActivity } from '../src/assistant.ts'
import { MODES } from '../src/modes.ts'

const parent: TaskSummary = {
  id: 'assistant-parent', kind: 'chat', target: 'conversation', label: 'Assistant',
  title: 'Check a proof', state: 'running', error: null, result_status: null,
  started: 1, finished: null,
}
const child: TranscriptItem = {
  role: 'route', content: '', time: '', mode: 'prove', orchestrated: true,
  task: parent.id, status: 'started', job_id: 'proof-child',
}

test('Square Harness preserves the free mode identifier for saved conversations', () => {
  assert.equal(MODES.free.label, 'Square Harness')
  assert.equal(MODES.free.mode, 'free')
})

test('a finished child is not running while its parent continues', () => {
  assert.deepEqual(routeActivity({ ...child, status: 'done' }, { task: parent, queue: [] }), { queued: false, running: false })
  for (const status of ['stopped', 'failed', 'cancelled', 'dismissed'] as const) {
    assert.deepEqual(routeActivity({ ...child, status }, { task: parent, queue: [parent] }), { queued: false, running: false })
  }
})

test('an active child follows its parent through running and pausing', () => {
  for (const state of ['starting', 'running', 'pausing'] as const) {
    assert.deepEqual(routeActivity(child, { task: { ...parent, state }, queue: [] }), { queued: false, running: true })
  }
  assert.deepEqual(routeActivity(child, { task: { ...parent, state: 'paused' }, queue: [] }), { queued: false, running: false })
})

test('queued child status and legacy route cards still use the task queue', () => {
  assert.deepEqual(routeActivity(child, { task: null, queue: [parent] }), { queued: true, running: false })
  assert.deepEqual(routeActivity({ ...child, orchestrated: undefined }, { task: parent, queue: [] }), { queued: false, running: true })
})

test('only paused Assistant turns are resumable', () => {
  assert.equal(assistantResumable({ mode: 'free', assistant_status: 'paused' }), true)
  assert.equal(assistantResumable({ mode: 'critic', assistant_status: 'paused' }), false)
  assert.equal(assistantResumable({ mode: 'free', assistant_status: 'complete' }), false)
  assert.equal(assistantResumable({ mode: 'free', assistant_status: 'incomplete' }), false)
  assert.equal(assistantResumable(null), false)
})

test('resumeAssistant targets the saved parent workflow without starting another child', async () => {
  const original = globalThis.fetch
  let call: { path: string; options?: RequestInit } | undefined
  globalThis.fetch = async (input, options) => {
    call = { path: String(input), options }
    return new Response(JSON.stringify({ task: parent }), { status: 200, headers: { 'Content-Type': 'application/json' } })
  }
  try {
    const result = await api.resumeAssistant('conversation')
    assert.equal(call?.path, '/api/chats/conversation/assistant/resume')
    assert.equal(call?.options?.method, 'POST')
    assert.equal(call?.options?.body, '{}')
    assert.equal(result.task.id, parent.id)
  } finally {
    globalThis.fetch = original
  }
})
