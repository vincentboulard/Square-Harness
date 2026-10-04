import { test } from 'node:test'
import assert from 'node:assert/strict'
import { api, ApiError, type TaskSnapshot, type TaskSummary } from '../src/api.ts'
import { chatDeleteBlocked, deletedChatDestination, withoutDeletedChat } from '../src/chatDeletion.ts'

const chat = 'conversation'
const task: TaskSummary = {
  id: 'turn', kind: 'chat', target: chat, label: 'Assistant', title: 'A proof',
  state: 'running', started: 1, finished: null, error: null, result_status: null,
}
const empty: TaskSnapshot = { seq: 1, task: null, activity: [], live: null, approvals: [], queue: [], routing: [] }

test('routing, active chat work and queued chat work prevent deletion', () => {
  assert.equal(chatDeleteBlocked({ ...empty, routing: [chat] }, chat), true)
  for (const state of ['queued', 'starting', 'running', 'pausing'] as const) {
    assert.equal(chatDeleteBlocked({ ...empty, task: { ...task, state } }, chat), true)
    assert.equal(chatDeleteBlocked({ ...empty, queue: [{ ...task, state }] }, chat), true)
  }
  assert.equal(chatDeleteBlocked({ ...empty, task: { ...task, target: 'other' } }, chat), false)
})

test('legacy proof and research jobs protect their originating conversation', () => {
  for (const kind of ['proof', 'research'] as const) {
    const child = { ...task, kind, target: 'worker-job', origin_chat: chat }
    assert.equal(chatDeleteBlocked({ ...empty, task: child }, chat), true)
    assert.equal(chatDeleteBlocked({ ...empty, queue: [{ ...child, state: 'queued' }] }, chat), true)
    assert.equal(chatDeleteBlocked({ ...empty, task: { ...child, state: 'done' } }, chat), false)
  }
})

test('live chat identity protects active output but permits deleting completed or paused turns', () => {
  const live = { chat, kind: 'message' as const, user: 'Prove it', steps: [] }
  assert.equal(chatDeleteBlocked({ ...empty, task: { ...task, target: null }, live }, chat), true)
  for (const state of ['paused', 'done', 'error', 'cancelled'] as const) {
    assert.equal(chatDeleteBlocked({ ...empty, task: { ...task, state }, live }, chat), false)
  }
})

test('deleted chat cleanup clears its finished turn and preserves an unrelated job', () => {
  const snapshot: TaskSnapshot = {
    ...empty, task: { ...task, kind: 'proof', target: 'job', origin_chat: chat, state: 'done' },
    live: { chat, kind: 'message', user: '', steps: [] },
    activity: [{ kind: 'notice', text: 'Finished', time: 1 }],
    approvals: [{ id: 'old', task: task.id, task_kind: 'chat', target: chat, kind: 'confirm', preview: '' }],
    queue: [{ ...task, id: 'other', target: 'other', state: 'queued' }], routing: [chat, 'other'],
  }
  const result = withoutDeletedChat(snapshot, chat)
  assert.equal(result.task, null)
  assert.equal(result.live, null)
  assert.deepEqual(result.activity, [])
  assert.deepEqual(result.approvals, [])
  assert.deepEqual(result.queue, snapshot.queue)
  assert.deepEqual(result.routing, ['other'])
  assert.deepEqual(withoutDeletedChat(result, chat), result)
  const unrelated = { ...empty, task: { ...task, target: 'other', state: 'done' as const } }
  assert.equal(withoutDeletedChat(unrelated, chat).task, unrelated.task)
})

test('only the currently selected deleted conversation returns to a new Assistant chat', () => {
  for (const mode of ['free', 'critic', 'explore']) {
    assert.equal(deletedChatDestination(`#/${mode}/${chat}`, chat), '#/free')
    assert.equal(deletedChatDestination(`#/${mode}/${chat}/context`, chat), '#/free')
    assert.equal(deletedChatDestination(`#/${mode}/other`, chat), null)
  }
  assert.equal(deletedChatDestination('#/free', chat), null)
  assert.equal(deletedChatDestination(`#/prove/${chat}`, chat), null)
})

test('deleteChat uses DELETE and exposes backend busy errors', async () => {
  const original = globalThis.fetch
  let call: { path: string; options?: RequestInit } | undefined
  let busy = false
  globalThis.fetch = async (input, options) => {
    call = { path: String(input), options }
    return new Response(JSON.stringify(busy ? { error: 'Pause this conversation before deleting it.' } : { deleted: chat }), {
      status: busy ? 409 : 200, headers: { 'Content-Type': 'application/json' },
    })
  }
  try {
    assert.deepEqual(await api.deleteChat(chat), { deleted: chat })
    assert.equal(call?.path, `/api/chats/${chat}`)
    assert.equal(call?.options?.method, 'DELETE')
    assert.equal(call?.options?.body, undefined)
    busy = true
    await assert.rejects(api.deleteChat(chat), (error: unknown) => error instanceof ApiError && error.status === 409 && error.message.includes('Pause'))
  } finally { globalThis.fetch = original }
})
