// Side panels the reader can slide away on wide screens: the notebook's list on the
// left and a job's overview (budget, candidates, checks) on the right. Remembered
// per browser; phones and narrow windows ignore it.
import { useSyncExternalStore } from 'react'

export type Panel = 'sidebar' | 'aside'

const KEYS: Record<Panel, string> = { sidebar: 'square-hide-sidebar', aside: 'square-hide-aside' }
const listeners = new Set<() => void>()

function read(panel: Panel) {
  try {
    return localStorage.getItem(KEYS[panel]) === '1'
  } catch {
    return false
  }
}

const hidden: Record<Panel, boolean> = { sidebar: read('sidebar'), aside: read('aside') }

export function setPanelHidden(panel: Panel, value: boolean) {
  hidden[panel] = value
  try {
    if (value) localStorage.setItem(KEYS[panel], '1')
    else localStorage.removeItem(KEYS[panel])
  } catch {
    /* private mode: the choice lasts for this page */
  }
  listeners.forEach((listener) => listener())
}

function subscribe(listener: () => void) {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export function usePanelHidden(panel: Panel): boolean {
  return useSyncExternalStore(subscribe, () => hidden[panel])
}

/** For event handlers: whether the panel is hidden right now. */
export function isPanelHidden(panel: Panel): boolean {
  return hidden[panel]
}
