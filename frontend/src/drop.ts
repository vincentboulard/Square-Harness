// One drop target at a time: the form or composer on screen decides what a dropped
// file is for (a pinned source, a template, an attachment). Files are always saved
// in the workspace first.
import { useEffect, useRef } from 'react'

type Handler = (paths: string[]) => void
let current: { handler: Handler; label: string } | null = null

export function useDropTarget(label: string, handler: Handler) {
  const latest = useRef(handler)
  latest.current = handler
  useEffect(() => {
    const entry = { handler: (paths: string[]) => latest.current(paths), label }
    current = entry
    return () => {
      if (current === entry) current = null
    }
  }, [label])
}

export function dropTarget() {
  return current
}
