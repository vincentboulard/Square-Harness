import { taskById } from './concurrency.ts'
import type { Chat, TaskSnapshot, TranscriptItem } from './api'

/** Delegated cards share their parent's task ID, even after the child returns. */
export function routeActivity(item: TranscriptItem, snapshot: Pick<TaskSnapshot, 'task' | 'queue'> & Partial<Pick<TaskSnapshot, 'tasks'>>) {
  const active = !item.orchestrated || ['starting', 'queued', 'started'].includes(item.status || '')
  const queued = active && !!item.task && snapshot.queue.some((task) => task.id === item.task)
  const running = active && !!taskById(snapshot, item.task)
  return { queued, running }
}

export const assistantResumable = (chat: Pick<Chat, 'mode' | 'assistant_status'> | null) =>
  chat?.mode === 'free' && chat.assistant_status === 'paused'
