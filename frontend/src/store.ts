// Application state fed by one Server-Sent Events connection.
import { useSyncExternalStore } from 'react'
import {
  api, ApiError, onUnauthorized,
  type ChatListItem, type LiveStep, type ProofListItem, type ResearchListItem,
  type Status, type StreamChunk, type TaskSnapshot, type TaskSummary, type Approval, type Activity,
} from './api'

export type Connection = 'connecting' | 'open' | 'reconnecting'

export type AppState = {
  auth: 'checking' | 'ok' | 'needed'
  status: Status | null
  connection: Connection
  snapshot: TaskSnapshot
  proofs: ProofListItem[] | null
  research: ResearchListItem[] | null
  chats: ChatListItem[] | null
  ticks: Record<string, number>
}

type ServerEvent =
  | { type: 'hello'; seq: number; instance: string; resync: boolean }
  | { type: 'task'; seq: number; task: TaskSummary }
  | ({ type: 'activity'; seq: number; task: string; job: string; id: string | null } & Activity)
  | { type: 'approval'; seq: number; state: 'pending' | 'approved' | 'denied'; approval: Approval | { id: string; task: string } }
  | { type: 'chat'; seq: number; chat: string; task: string; kind: 'start' | 'delta' | 'end' | 'tool' | 'result' | 'notice'; thinking?: string; text?: string }
  | { type: 'chat_saved'; seq: number; chat: string }
  | { type: 'job'; seq: number; job: 'proof' | 'research'; id: string; status: string; phase: string | null }
  | ({ type: 'stream'; seq: number; job: 'proof' | 'research'; id: string; role: string } & StreamChunk)
  | { type: 'queue'; seq: number; queue: TaskSummary[] }
  | { type: 'routing'; seq: number; chat: string; state: 'running' | 'done' }
  | { type: 'workspace'; seq: number; path: string }
  | { type: 'files'; seq: number; path: string }

const EMPTY: TaskSnapshot = { seq: 0, task: null, activity: [], live: null, approvals: [], queue: [], routing: [] }

let state: AppState = {
  auth: 'checking', status: null, connection: 'connecting', snapshot: EMPTY,
  proofs: null, research: null, chats: null, ticks: {},
}
const listeners = new Set<() => void>()

function set(patch: Partial<AppState> | ((s: AppState) => Partial<AppState>)) {
  state = { ...state, ...(typeof patch === 'function' ? patch(state) : patch) }
  listeners.forEach((listener) => listener())
}

function subscribe(listener: () => void) {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export function useApp(): AppState {
  return useSyncExternalStore(subscribe, () => state)
}

export function getState() {
  return state
}

/** Changes whenever the server reports a change to this job or chat. */
export function useTick(id: string | null | undefined): number {
  const app = useApp()
  return id ? app.ticks[id] || 0 : 0
}

function bump(id: string | null | undefined) {
  if (!id) return
  set((s) => ({ ticks: { ...s.ticks, [id]: (s.ticks[id] || 0) + 1 } }))
}

// -- live model output ------------------------------------------------------

type StreamListener = (chunk: StreamChunk & { role: string }) => void
const streamListeners = new Map<string, Set<StreamListener>>()

export function onStream(job: string, id: string, file: string, listener: StreamListener) {
  const key = `${job}:${id}:${file}`
  if (!streamListeners.has(key)) streamListeners.set(key, new Set())
  streamListeners.get(key)!.add(listener)
  return () => {
    streamListeners.get(key)?.delete(listener)
  }
}

// -- lists ---------------------------------------------------------------------

let listTimer: number | undefined
export function refreshLists(delay = 0) {
  window.clearTimeout(listTimer)
  listTimer = window.setTimeout(async () => {
    try {
      const [proofs, research, chats] = await Promise.all([api.proofs(), api.researches(), api.chats()])
      set({ proofs: proofs.jobs, research: research.jobs, chats: chats.chats })
    } catch (error) {
      if (!(error instanceof ApiError && error.status === 401)) console.warn(error)
    }
  }, delay)
}

export async function refreshStatus() {
  try {
    set({ status: await api.status() })
  } catch {
    /* connection state already tells the user */
  }
}

// -- task snapshot and events ------------------------------------------------

let buffered: ServerEvent[] | null = null

async function refreshSnapshot() {
  buffered = buffered || []
  try {
    const snapshot = await api.task()
    const pending = buffered || []
    buffered = null
    set({ snapshot })
    pending.filter((event) => event.seq > snapshot.seq).forEach(handle)
  } catch {
    buffered = null
  }
}

function applyChat(event: Extract<ServerEvent, { type: 'chat' }>) {
  set((s) => {
    const live = s.snapshot.live
    if (!live || s.snapshot.task?.id !== event.task) return {}
    const steps: LiveStep[] = [...live.steps]
    if (event.kind === 'start') steps.push({ type: 'call', thinking: '', text: '' })
    else if (event.kind === 'delta') {
      const last = steps[steps.length - 1]
      if (last && last.type === 'call') {
        steps[steps.length - 1] = { ...last, thinking: last.thinking + (event.thinking || ''), text: last.text + (event.text || '') }
      }
    } else if (event.kind === 'tool' || event.kind === 'result' || event.kind === 'notice') {
      steps.push({ type: event.kind, text: event.text || '' })
    }
    return { snapshot: { ...s.snapshot, seq: event.seq, live: { ...live, steps } } }
  })
}

let instance = ''

function handle(event: ServerEvent) {
  // A task event (such as a new job's ID) may be newer than the snapshot being fetched.
  if (buffered && (event.type === 'chat' || event.type === 'activity' || event.type === 'approval' || event.type === 'task')) {
    buffered.push(event)
    return
  }
  switch (event.type) {
    case 'hello':
      if (event.resync || (instance && instance !== event.instance)) {
        refreshAll()
      }
      instance = event.instance
      break
    case 'task': {
      const current = state.snapshot.task
      const fresh = !current || current.id !== event.task.id
      // A job joining or leaving the queue is not the model's task: keep the running one shown.
      const waiting = event.task.state === 'queued' || event.task.state === 'cancelled'
      if (!(fresh && waiting)) {
        set((s) => ({
          snapshot: fresh
            ? { ...s.snapshot, seq: event.seq, task: event.task, activity: [], live: null }
            : { ...s.snapshot, seq: event.seq, task: event.task },
        }))
      }
      if (fresh && event.task.state === 'running') refreshSnapshot()
      if (['done', 'paused', 'error'].includes(event.task.state)) {
        refreshLists(150)
        bump(event.task.target)
      } else if (event.task.target) {
        bump(event.task.target)
        refreshLists(300)
      }
      break
    }
    case 'activity':
      set((s) => s.snapshot.task?.id === event.task
        ? { snapshot: { ...s.snapshot, activity: [...s.snapshot.activity.slice(-199), { kind: event.kind, text: event.text, time: event.time }] } }
        : {})
      break
    case 'approval':
      set((s) => ({
        snapshot: {
          ...s.snapshot,
          approvals: event.state === 'pending'
            ? [...s.snapshot.approvals.filter((a) => a.id !== event.approval.id), event.approval as Approval]
            : s.snapshot.approvals.filter((a) => a.id !== event.approval.id),
        },
      }))
      break
    case 'chat':
      applyChat(event)
      break
    case 'chat_saved':
      refreshLists(100)
      bump(event.chat)
      break
    case 'job':
      refreshLists(400)
      bump(event.id)
      break
    case 'stream':
      streamListeners.get(`${event.job}:${event.id}:${event.file}`)?.forEach((listener) => listener(event))
      break
    case 'queue':
      set((s) => ({ snapshot: { ...s.snapshot, queue: event.queue } }))
      break
    case 'routing':
      set((s) => ({
        snapshot: {
          ...s.snapshot,
          routing: event.state === 'running' ? [...new Set([...s.snapshot.routing, event.chat])] : s.snapshot.routing.filter((c) => c !== event.chat),
        },
      }))
      bump(event.chat)
      break
    case 'workspace':
      // Jobs and chats belong to a folder: leave any page from the previous one.
      window.location.hash = window.location.hash.split('/').slice(0, 2).join('/') || '#/free'
      refreshAll()
      break
    case 'files':
      bump('files')
      break
  }
}

// -- connection ------------------------------------------------------------------

let source: EventSource | null = null
let retry: number | undefined

function connect() {
  source?.close()
  window.clearTimeout(retry)
  source = new EventSource('/api/events')
  source.onopen = () => set({ connection: 'open' })
  source.onmessage = (message) => {
    try {
      handle(JSON.parse(message.data))
    } catch (error) {
      console.warn('Ignored malformed event', error)
    }
  }
  source.onerror = () => {
    set({ connection: 'reconnecting' })
    if (source && source.readyState === EventSource.CLOSED) {
      // The browser gave up (server stopped, or the token was refused).
      retry = window.setTimeout(async () => {
        try {
          await api.status()
          connect()
          refreshAll()
        } catch (error) {
          if (error instanceof ApiError && error.status === 401) return
          retry = window.setTimeout(() => connect(), 3000)
        }
      }, 2000)
    }
  }
}

export function refreshAll() {
  refreshStatus()
  refreshLists()
  refreshSnapshot()
  set((s) => ({ ticks: Object.fromEntries(Object.entries(s.ticks).map(([k, v]) => [k, v + 1])) }))
}

export async function start() {
  onUnauthorized(() => {
    source?.close()
    set({ auth: 'needed' })
  })
  try {
    const status = await api.status()
    set({ status, auth: 'ok' })
    connect()
    refreshLists()
    refreshSnapshot()
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) set({ auth: 'needed' })
    else {
      set({ auth: 'ok', connection: 'reconnecting' })
      retry = window.setTimeout(start, 3000)
    }
  }
}

export async function login(token: string) {
  await api.login(token)
  await start()
}

// -- theme ---------------------------------------------------------------------------

export type Theme = 'system' | 'light' | 'dark'

export function readTheme(): Theme {
  try {
    const value = localStorage.getItem('square-theme')
    return value === 'light' || value === 'dark' ? value : 'system'
  } catch {
    return 'system'
  }
}

export function applyTheme(theme: Theme) {
  if (theme === 'system') delete document.documentElement.dataset.theme
  else document.documentElement.dataset.theme = theme
  try {
    localStorage.setItem('square-theme', theme)
  } catch {
    /* private mode: the choice lasts for this page */
  }
}
