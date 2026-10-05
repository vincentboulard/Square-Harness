import { test } from 'node:test'
import assert from 'node:assert/strict'
import { api, type TaskEntry, type TaskSnapshot, type TaskSummary } from '../src/api.ts'
import { activeTasks, capacityFull, chatWork, reduceActivity, reduceChat, reduceTask, taskForJob, workspaceBusy } from '../src/concurrency.ts'
import { chatDeleteBlocked } from '../src/chatDeletion.ts'
import { routeActivity } from '../src/assistant.ts'

const task = (id: string, target: string): TaskSummary => ({
  id, target, kind: 'chat', label: 'Assistant', title: target, state: 'running',
  started: 1, finished: null, error: null, result_status: null,
})
const entry = (id: string, chat: string): TaskEntry => ({
  task: task(id, chat), activity: [], live: { chat, kind: 'message', user: chat, steps: [] },
})
const first = entry('first', 'chat-a')
const second = entry('second', 'chat-b')
const snapshot: TaskSnapshot = {
  seq: 1, task: first.task, activity: [], live: first.live, tasks: [first, second],
  approvals: [], queue: [], routing: [], concurrency: { chats: 2, requests: 2, active: 2 },
}

test('interleaved chat deltas and activity stay attached to their own task after reconnect', () => {
  let current = { ...snapshot, tasks: JSON.parse(JSON.stringify(snapshot.tasks)) }
  current = reduceChat(current, { task: 'first', chat: 'chat-a', seq: 2, kind: 'start' })
  current = reduceChat(current, { task: 'second', chat: 'chat-b', seq: 3, kind: 'start' })
  current = reduceChat(current, { task: 'first', chat: 'chat-a', seq: 4, kind: 'delta', text: 'Alpha' })
  current = reduceChat(current, { task: 'second', chat: 'chat-b', seq: 5, kind: 'delta', thinking: 'Beta thought', text: 'Beta' })
  current = reduceActivity(current, 'second', { kind: 'tool', text: 'Read b.tex', time: 1 }, 6)
  // Reconnect replaces state with a server snapshot before later events continue.
  current = JSON.parse(JSON.stringify(current))
  current = reduceChat(current, { task: 'first', chat: 'chat-a', seq: 7, kind: 'delta', text: ' continued' })
  assert.deepEqual(chatWork(current, 'chat-a').active?.live?.steps, [{ type: 'call', text: 'Alpha continued', thinking: '' }])
  assert.deepEqual(chatWork(current, 'chat-b').active?.live?.steps, [{ type: 'call', text: 'Beta', thinking: 'Beta thought' }])
  assert.deepEqual(chatWork(current, 'chat-a').active?.activity, [])
  assert.equal(chatWork(current, 'chat-b').active?.activity[0].text, 'Read b.tex')
  assert.deepEqual(snapshot.tasks?.[0].live?.steps, [])
})

test('completion and pausing affect only the named task, including the legacy snapshot fields', () => {
  let current = reduceTask(snapshot, { ...second.task, state: 'pausing' }, 2)
  assert.equal(chatWork(current, 'chat-a').active?.task.state, 'running')
  assert.equal(chatWork(current, 'chat-b').active?.task.state, 'pausing')
  current = reduceTask(current, { ...first.task, state: 'done' }, 3)
  assert.equal(chatWork(current, 'chat-a').blocked, false)
  assert.equal(chatWork(current, 'chat-b').blocked, true)
  assert.equal(current.task?.id, 'second')
  assert.equal(current.live?.chat, 'chat-b')
  assert.equal(activeTasks(current).length, 1)
  assert.equal(capacityFull(current), false)
})

test('late or mismatched events cannot put another conversation output into a live turn', () => {
  const current = reduceChat(snapshot, { task: 'first', chat: 'chat-b', seq: 2, kind: 'notice', text: 'wrong chat' })
  assert.deepEqual(current.tasks?.map((item) => item.live?.steps), [[], []])
  const finished = reduceTask(current, { ...first.task, state: 'paused' }, 3)
  const late = reduceChat(finished, { task: 'first', chat: 'chat-a', seq: 4, kind: 'notice', text: 'late' })
  assert.equal(activeTasks(late).length, 1)
  assert.equal(chatWork(late, 'chat-b').active?.task.id, 'second')
  assert.deepEqual(late.live?.steps, [])
})

test('one active or queued turn blocks only its own conversation while the third can queue', () => {
  assert.equal(chatWork({ ...snapshot, tasks: [{ ...first, task: { ...first.task, origin_chat: null } }] }, null).blocked, false)
  assert.equal(chatWork(snapshot, 'chat-a').blocked, true)
  assert.equal(chatWork(snapshot, 'chat-b').blocked, true)
  assert.equal(chatWork(snapshot, 'chat-c').blocked, false)
  assert.equal(capacityFull(snapshot), true)
  const queued = { ...snapshot, queue: [{ ...task('third', 'chat-c'), state: 'queued' as const }] }
  assert.equal(chatWork(queued, 'chat-c').blocked, true)
  assert.equal(chatWork(queued, 'chat-c').queued?.id, 'third')
  assert.equal(chatWork(queued, 'chat-d').blocked, false)
  assert.equal(chatDeleteBlocked(snapshot, 'chat-b'), true)
  assert.equal(chatDeleteBlocked(queued, 'chat-c'), true)
  assert.equal(chatDeleteBlocked(queued, 'chat-d'), false)
})

test('all active work and queued work protect folder changes even when legacy task is finished', () => {
  assert.equal(workspaceBusy({ ...snapshot, task: { ...first.task, state: 'done' } }), true)
  const empty = { ...snapshot, task: { ...first.task, state: 'done' as const }, tasks: [] }
  assert.equal(workspaceBusy(empty), false)
  assert.equal(workspaceBusy({ ...empty, queue: [{ ...second.task, state: 'queued' }] }), true)
  assert.equal(workspaceBusy({ ...empty, routing: ['chat-c'] }), true)
})

test('legacy snapshots still work and a new empty active list overrides stale legacy data', () => {
  assert.equal(activeTasks({ ...snapshot, tasks: undefined }).length, 1)
  assert.equal(activeTasks({ ...snapshot, tasks: [] }).length, 0)
  assert.equal(taskForJob(snapshot, 'chat', 'chat-b')?.task.id, 'second')
  assert.deepEqual(routeActivity({ role: 'route', content: '', time: '', task: 'second', status: 'started', orchestrated: true }, snapshot), { queued: false, running: true })
})

test('pause, limits, sends and reviews use explicit task identity and automatic queueing', async () => {
  const original = globalThis.fetch
  const calls: { path: string; body: unknown }[] = []
  globalThis.fetch = async (input, options) => {
    calls.push({ path: String(input), body: JSON.parse(String(options?.body)) })
    return new Response(JSON.stringify({ task: second.task, concurrency: { chats: 3, requests: 2, active: 2 } }), { status: 200 })
  }
  try {
    await api.pause('second')
    await api.concurrency({ chats: 3, requests: 2 })
    await api.send('chat-c', 'Another question')
    await api.review('chat-c')
    assert.deepEqual(calls, [
      { path: '/api/task/pause', body: { task: 'second' } },
      { path: '/api/concurrency', body: { chats: 3, requests: 2 } },
      { path: '/api/chats/chat-c/messages', body: { content: 'Another question', files: [], queue: true } },
      { path: '/api/chats/chat-c/review', body: { queue: true } },
    ])
  } finally { globalThis.fetch = original }
})
