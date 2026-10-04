import type { Chat, TaskSnapshot, TranscriptItem } from './api'

/** Delegated cards share their parent's task ID, even after the child returns. */
export function routeActivity(item: TranscriptItem, snapshot: Pick<TaskSnapshot, 'task' | 'queue'>) {
  const active = !item.orchestrated || ['starting', 'queued', 'started'].includes(item.status || '')
  const queued = active && !!item.task && snapshot.queue.some((task) => task.id === item.task)
  const running = active && !!item.task && snapshot.task?.id === item.task
    && ['starting', 'running', 'pausing'].includes(snapshot.task.state)
  return { queued, running }
}

export const assistantResumable = (chat: Pick<Chat, 'mode' | 'assistant_status'> | null) =>
  chat?.mode === 'free' && chat.assistant_status === 'paused'
