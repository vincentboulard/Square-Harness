import { useSyncExternalStore } from 'react'

export type Mode = 'free' | 'prove' | 'critic' | 'explore' | 'literature' | 'referee' | 'writeup'
export const MODE_ORDER: Mode[] = ['free', 'prove', 'critic', 'explore', 'literature', 'referee', 'writeup']
const UUID = /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/

export type Route = { mode: Mode; id: string | null; tab: string | null; from: string | null }

export function parse(hash: string): Route {
  const [path, search = ''] = hash.replace(/^#\/?/, '').split('?')
  const [first, second, third] = path.split('/')
  const requestedMode = (MODE_ORDER as string[]).includes(first) ? (first as Mode) : 'free'
  const id = second && UUID.test(second) ? second : null
  const mode = id ? requestedMode : 'free'
  const origin = new URLSearchParams(search).get('from')
  const from = id && origin && UUID.test(origin) ? origin : null
  return { mode, id, tab: id && third ? third : null, from }
}

function subscribe(callback: () => void) {
  window.addEventListener('hashchange', callback)
  return () => window.removeEventListener('hashchange', callback)
}

export function useRoute(): Route {
  const hash = useSyncExternalStore(subscribe, () => window.location.hash)
  return parse(hash)
}

export function href(mode: Mode, id?: string | null, tab?: string | null, from?: string | null) {
  const path = '#/' + [mode, id, id ? tab : null].filter(Boolean).join('/')
  return path + (id && from && UUID.test(from) ? '?from=' + from : '')
}

export function go(mode: Mode, id?: string | null, tab?: string | null, from?: string | null) {
  const current = parse(window.location.hash)
  const origin = from === undefined && current.mode === mode && current.id === id ? current.from : from
  window.location.hash = href(mode, id, tab, origin)
}

/** The notebook that holds a mode's pages: older critic and explore chats live in Default. */
export function notebook(mode: Mode): Mode {
  return mode === 'critic' || mode === 'explore' ? 'free' : mode
}
