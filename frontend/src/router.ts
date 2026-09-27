import { useSyncExternalStore } from 'react'

export type Mode = 'free' | 'prove' | 'critic' | 'explore' | 'literature' | 'referee' | 'writeup'
export const MODE_ORDER: Mode[] = ['free', 'prove', 'critic', 'explore', 'literature', 'referee', 'writeup']
// The notebooks in the rail. Critic and explore remain workflows of the engine (the
// Default notebook answers with them) and their older conversations stay readable.
export const RAIL: Mode[] = ['free', 'prove', 'literature', 'referee', 'writeup']
const UUID = /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/

export type Page = 'about'
export type Route = { mode: Mode; id: string | null; tab: string | null; page: Page | null }

export function parse(hash: string): Route {
  const [first, second, third] = hash.replace(/^#\/?/, '').split('/')
  if (first === 'about') return { mode: 'free', id: null, tab: null, page: 'about' }
  const mode = (MODE_ORDER as string[]).includes(first) ? (first as Mode) : 'free'
  const id = second && UUID.test(second) ? second : null
  return { mode, id, tab: id && third ? third : null, page: null }
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

/** The notebook that holds a mode's pages: older critic and explore chats live in Default. */
export function notebook(mode: Mode): Mode {
  return mode === 'critic' || mode === 'explore' ? 'free' : mode
}
