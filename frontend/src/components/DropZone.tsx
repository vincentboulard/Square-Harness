import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { dropTarget } from '../drop'
import { useApp } from '../store'

type Toast = { id: number; text: string; error?: boolean }

/** Drag files anywhere over the work pane: they are saved in the workspace. */
export function DropZone() {
  const { status } = useApp()
  const [over, setOver] = useState(false)
  const [toasts, setToasts] = useState<Toast[]>([])
  const depth = useRef(0)
  const counter = useRef(0)

  useEffect(() => {
    const hasFiles = (event: DragEvent) => Array.from(event.dataTransfer?.types || []).includes('Files')
    const toast = (text: string, error = false) => {
      const id = ++counter.current
      setToasts((items) => [...items, { id, text, error }])
      window.setTimeout(() => setToasts((items) => items.filter((item) => item.id !== id)), error ? 7000 : 4000)
    }
    const enter = (event: DragEvent) => {
      if (!hasFiles(event)) return
      event.preventDefault()
      depth.current += 1
      setOver(true)
    }
    const leave = (event: DragEvent) => {
      if (!hasFiles(event)) return
      depth.current = Math.max(0, depth.current - 1)
      if (!depth.current) setOver(false)
    }
    const overHandler = (event: DragEvent) => {
      if (!hasFiles(event)) return
      event.preventDefault()
      if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy'
    }
    const drop = async (event: DragEvent) => {
      if (!hasFiles(event)) return
      event.preventDefault()
      depth.current = 0
      setOver(false)
      const files = Array.from(event.dataTransfer?.files || [])
      const saved: string[] = []
      for (const file of files) {
        try {
          const result = await api.upload(file)
          saved.push(result.path)
          toast(result.path === file.name ? `Added ${result.path}` : `Added ${file.name} as ${result.path} (a file with that name exists)`)
        } catch (reason) {
          toast(`${file.name}: ${(reason as Error).message}`, true)
        }
      }
      if (saved.length) dropTarget()?.handler(saved)
    }
    window.addEventListener('dragenter', enter)
    window.addEventListener('dragleave', leave)
    window.addEventListener('dragover', overHandler)
    window.addEventListener('drop', drop)
    return () => {
      window.removeEventListener('dragenter', enter)
      window.removeEventListener('dragleave', leave)
      window.removeEventListener('dragover', overHandler)
      window.removeEventListener('drop', drop)
    }
  }, [])

  const folder = status?.workspace.split('/').filter(Boolean).pop() || 'the workspace'
  const target = dropTarget()
  return (
    <>
      {over && (
        <div className="drop-overlay" aria-hidden="true">
          <div className="drop-card">
            <p className="drop-title">Drop to add to {folder}</p>
            <p className="drop-hint">{target ? target.label : 'Files are saved in the folder; nothing is replaced.'}</p>
            <p className="drop-types">.tex .sty .cls .bib .md .txt .pdf .py</p>
          </div>
        </div>
      )}
      <div className="toasts" role="status">
        {toasts.map((item) => <p key={item.id} className={'toast' + (item.error ? ' toast-error' : '')}>{item.text}</p>)}
      </div>
    </>
  )
}
