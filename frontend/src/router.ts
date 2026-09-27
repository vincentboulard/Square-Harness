import { useSyncExternalStore } from 'react'

export type Mode = 'free' | 'prove' | 'critic' | 'explore' | 'literature' | 'referee' | 'writeup'
export const MODE_ORDER: Mode[] = ['free', 'prove', 'critic', 'explore', 'literature', 'referee', 'writeup']
const UUID = /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/

export type Route = { mode: Mode; id: string | null; tab: string | null }

export function parse(hash: string): Route {
  const [first, second, third] = hash.replace(/^#\/?/, '').split('/')
  const mode = (MODE_ORDER as string[]).includes(first) ? (first as Mode) : 'free'
  const id = second && UUID.test(second) ? second : null
  return { mode, id, tab: id && third ? third : null }
}

function subscribe(callback: () => void) {
  window.addEventListener('hashchange', callback)
  return () => window.removeEventListener('hashchange', callback)
}

export function useRoute(): Route {
  const hash = useSyncExternalStore(subscribe, () => window.location.hash)
  return parse(hash)
}

export function href(mode: Mode, id?: string | null, tab?: string | null) {
  return '#/' + [mode, id, id ? tab : null].filter(Boolean).join('/')
}

export function go(mode: Mode, id?: string | null, tab?: string | null) {
  window.location.hash = href(mode, id, tab)
}
