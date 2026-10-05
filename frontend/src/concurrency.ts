import type { Activity, Concurrency, LiveStep, TaskEntry, TaskSnapshot, TaskSummary } from './api.ts'

export const taskRunning = (task: TaskSummary | null | undefined) =>
  !!task && ['starting', 'running', 'pausing'].includes(task.state)

type TaskSource = Pick<TaskSnapshot, 'task'> & Partial<Pick<TaskSnapshot, 'tasks' | 'activity' | 'live'>>

/** New snapshots are authoritative, including an explicitly empty active list. */
export function activeTasks(snapshot: TaskSource): TaskEntry[] {
  if (snapshot.tasks !== undefined) return snapshot.tasks.filter((entry) => taskRunning(entry.task))
  return taskRunning(snapshot.task)
    ? [{ task: snapshot.task!, activity: snapshot.activity || [], live: snapshot.live || null }]
    : []
}

export const taskById = (snapshot: TaskSource, id: string | null | undefined) =>
  activeTasks(snapshot).find((entry) => entry.task.id === id)

export const taskForJob = (snapshot: TaskSource, kind: TaskSummary['kind'], id: string | null) =>
  activeTasks(snapshot).find((entry) => entry.task.kind === kind && entry.task.target === id)

export function chatWork(snapshot: TaskSnapshot, id: string | null) {
  if (!id) return { active: undefined, queued: undefined, routing: false, blocked: false }
  const active = activeTasks(snapshot).find((entry) => entry.task.target === id || entry.task.origin_chat === id || entry.live?.chat === id)
  const queued = snapshot.queue.find((task) => task.target === id || task.origin_chat === id)
  const routing = !!id && snapshot.routing.includes(id)
  return { active, queued, routing, blocked: !!active || !!queued || routing }
}

export const concurrencyLimits = (snapshot: TaskSnapshot, status?: { concurrency?: Concurrency } | null): Concurrency =>
  snapshot.concurrency || status?.concurrency || { chats: 2, requests: 2, active: activeTasks(snapshot).length }

export const capacityFull = (snapshot: TaskSnapshot, status?: { concurrency?: Concurrency } | null) =>
  activeTasks(snapshot).length >= concurrencyLimits(snapshot, status).chats || snapshot.queue.length > 0

/** Queued work also belongs to the folder it was submitted in. */
export const workspaceBusy = (snapshot: TaskSnapshot) =>
  activeTasks(snapshot).length > 0 || snapshot.queue.length > 0 || snapshot.routing.length > 0

function updateEntry(snapshot: TaskSnapshot, id: string, seq: number, update: (entry: TaskEntry) => TaskEntry): TaskSnapshot {
  const tasks = activeTasks(snapshot).map((entry) => entry.task.id === id ? update(entry) : entry)
  const legacy = tasks.find((entry) => entry.task.id === snapshot.task?.id)
  return { ...snapshot, tasks, seq: Math.max(seq, snapshot.seq), ...(legacy ? { task: legacy.task, activity: legacy.activity, live: legacy.live } : {}) }
}

export function reduceTask(snapshot: TaskSnapshot, task: TaskSummary, seq: number): TaskSnapshot {
  const tasks = activeTasks(snapshot).filter((entry) => entry.task.id !== task.id)
  const previous = taskById(snapshot, task.id)
  if (taskRunning(task)) tasks.push({ task, activity: previous?.activity || [], live: previous?.live || null })
  // Keep legacy consumers pointed at the same task until it leaves; new tasks never
  // replace another task's buffers. Terminal summaries remain available separately.
  const legacy = tasks.find((entry) => entry.task.id === snapshot.task?.id) || tasks[0]
  return {
    ...snapshot, tasks, seq: Math.max(seq, snapshot.seq),
    task: legacy?.task || task,
    activity: legacy?.activity || previous?.activity || [],
    live: legacy?.live || null,
  }
}

export function reduceActivity(snapshot: TaskSnapshot, task: string, activity: Activity, seq: number): TaskSnapshot {
  return updateEntry(snapshot, task, seq, (entry) => ({ ...entry, activity: [...entry.activity.slice(-199), activity] }))
}

export type ChatEvent = {
  seq: number; task: string; chat: string; kind: 'start' | 'delta' | 'end' | 'tool' | 'result' | 'notice'; thinking?: string; text?: string
}

export function reduceChat(snapshot: TaskSnapshot, event: ChatEvent): TaskSnapshot {
  return updateEntry(snapshot, event.task, event.seq, (entry) => {
    if (!entry.live || entry.live.chat !== event.chat) return entry
    const steps: LiveStep[] = [...entry.live.steps]
    if (event.kind === 'start') steps.push({ type: 'call', thinking: '', text: '' })
    else if (event.kind === 'delta') {
      const last = steps[steps.length - 1]
      if (last?.type === 'call') steps[steps.length - 1] = { ...last, thinking: last.thinking + (event.thinking || ''), text: last.text + (event.text || '') }
    } else if (event.kind === 'tool' || event.kind === 'result' || event.kind === 'notice') steps.push({ type: event.kind, text: event.text || '' })
    return { ...entry, live: { ...entry.live, steps } }
  })
}
