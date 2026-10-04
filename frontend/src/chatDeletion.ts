import type { TaskSnapshot, TaskSummary } from './api.ts'

const BUSY = new Set(['queued', 'starting', 'running', 'pausing'])

function belongsToChat(task: TaskSummary | null, id: string) {
  return !!task && (task.target === id || task.origin_chat === id)
}

/** A legacy delegated job can still be using its originating conversation. */
export function chatDeleteBlocked(snapshot: TaskSnapshot, id: string): boolean {
  if (snapshot.routing.includes(id)) return true
  if (snapshot.task && BUSY.has(snapshot.task.state)
    && (belongsToChat(snapshot.task, id) || snapshot.live?.chat === id)) return true
  return snapshot.queue.some((task) => BUSY.has(task.state) && belongsToChat(task, id))
}

/** Deletion also retires a finished turn that the task panel still remembers. */
export function withoutDeletedChat(snapshot: TaskSnapshot, id: string): TaskSnapshot {
  const removeTask = belongsToChat(snapshot.task, id) && !BUSY.has(snapshot.task!.state)
  return {
    ...snapshot,
    task: removeTask ? null : snapshot.task,
    live: snapshot.live?.chat === id ? null : snapshot.live,
    activity: removeTask ? [] : snapshot.activity,
    approvals: removeTask ? snapshot.approvals.filter((approval) => approval.task !== snapshot.task!.id) : snapshot.approvals,
    queue: snapshot.queue.filter((task) => !belongsToChat(task, id)),
    routing: snapshot.routing.filter((chat) => chat !== id),
  }
}

export function deletedChatDestination(hash: string, id: string): string | null {
  const [, mode, selected] = hash.split('/')
  return ['free', 'critic', 'explore'].includes(mode) && selected === id ? '#/free' : null
}
