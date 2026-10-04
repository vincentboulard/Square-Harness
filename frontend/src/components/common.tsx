import { useCallback, useEffect, useId, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { api, type WorkspaceFile } from '../api'
import { bytes, count } from '../format'
import { useTick } from '../store'
import { CloseIcon, FileIcon, PinIcon } from './Icons'

/** Load data whenever `deps` change; `reload` refetches without clearing. */
export function useLoad<T>(load: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState('')
  const latest = useRef(0)
  const run = useCallback(() => {
    const ticket = ++latest.current
    load().then(
      (value) => { if (ticket === latest.current) { setData(value); setError('') } },
      (reason: Error) => { if (ticket === latest.current) setError(reason.message) },
    )
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  useEffect(() => { run() }, [run])
  return { data, error, reload: run }
}

export function Meter({ label, used, limit, unit = '', decimals = 0 }: { label: string; used: number; limit: number; unit?: string; decimals?: number }) {
  const ratio = limit > 0 ? Math.min(1, used / limit) : 0
  const show = (value: number) => (decimals ? new Intl.NumberFormat(undefined, { maximumFractionDigits: decimals, minimumFractionDigits: decimals }).format(value) : count(value))
  const tone = ratio >= 1 ? ' meter-full' : ratio > 0.85 ? ' meter-high' : ''
  return (
    <div className={'meter' + tone}>
      <div className="meter-text">
        <span>{label}</span>
        <span className="meter-value">{show(used)}{unit} <span className="meter-of">of {show(limit)}{unit}</span></span>
      </div>
      <div className="meter-track" role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={limit} aria-valuenow={used}>
        <div className="meter-bar" style={{ transform: `scaleX(${ratio})` }} />
      </div>
    </div>
  )
}

export function Collapse({ summary, children, open, className }: { summary: ReactNode; children: ReactNode; open?: boolean; className?: string }) {
  return (
    <details className={'collapse' + (className ? ' ' + className : '')} open={open}>
      <summary>{summary}</summary>
      <div className="collapse-body">{children}</div>
    </details>
  )
}

export type Tab = { id: string; label: string; badge?: ReactNode; hidden?: boolean; narrowOnly?: boolean }

export function Tabs({ tabs, active, onChange }: { tabs: Tab[]; active: string; onChange: (id: string) => void }) {
  return (
    <div className="tabs" role="tablist">
      {tabs.filter((tab) => !tab.hidden).map((tab) => (
        <button key={tab.id} type="button" role="tab" aria-selected={tab.id === active}
          className={'tab' + (tab.id === active ? ' tab-active' : '') + (tab.narrowOnly ? ' tab-narrow' : '')}
          onClick={() => onChange(tab.id)}>
          {tab.label}{tab.badge !== undefined && <span className="tab-badge">{tab.badge}</span>}
        </button>
      ))}
    </div>
  )
}

export function NumberField({ label, value, onChange, min, max, step = 1, hint, suffix }: {
  label: string; value: number; onChange: (value: number) => void; min?: number; max?: number; step?: number; hint?: string; suffix?: string
}) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <span className="field-input">
        <input type="number" inputMode="numeric" value={Number.isFinite(value) ? value : ''} min={min} max={max} step={step}
          onChange={(event) => onChange(event.target.valueAsNumber)} />
        {suffix && <span className="field-suffix">{suffix}</span>}
      </span>
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  )
}

export function Toggle({ label, checked, onChange, disabled, hint }: {
  label: string; checked: boolean; onChange: (value: boolean) => void; disabled?: boolean; hint?: string
}) {
  return (
    <label className={'toggle' + (disabled ? ' toggle-disabled' : '')}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} />
      <span className="toggle-track" aria-hidden="true"><span className="toggle-thumb" /></span>
      <span className="toggle-text">
        <span>{label}</span>
        {hint && <span className="field-hint">{hint}</span>}
      </span>
    </label>
  )
}

export function FilePicker({ selected, onChange, allowPdf = false, accept, label = 'Pin workspace files' }: {
  selected: string[]; onChange: (files: string[]) => void; allowPdf?: boolean; accept?: string[]; label?: string
}) {
  const [open, setOpen] = useState(false)
  const [files, setFiles] = useState<WorkspaceFile[] | null>(null)
  const [error, setError] = useState('')
  const changed = useTick('files')
  useEffect(() => { setFiles(null) }, [changed])  // a dropped file or another folder
  useEffect(() => {
    if (!open || files) return
    api.files().then((result) => setFiles(result.files), (reason: Error) => setError(reason.message))
  }, [open, files])
  const available = (files || []).filter((file) => (allowPdf || file.kind === 'text')
    && (!accept || accept.some((suffix) => file.path.toLowerCase().endsWith(suffix))))
  const toggle = (path: string) =>
    onChange(selected.includes(path) ? selected.filter((item) => item !== path) : [...selected, path])
  return (
    <div className="picker">
      <div className="picker-row">
        {selected.map((path) => (
          <span key={path} className="chip">
            <FileIcon size={14} />
            <span>{path}</span>
            <button type="button" className="chip-remove" aria-label={`Unpin ${path}`} onClick={() => toggle(path)}>
              <CloseIcon size={13} />
            </button>
          </span>
        ))}
        <button type="button" className="btn btn-quiet" onClick={() => setOpen(!open)} aria-expanded={open}>
          <PinIcon size={16} /> {open ? 'Done' : label}
        </button>
      </div>
      {open && (
        <div className="picker-list">
          {error && <p className="error-note">{error}</p>}
          {!files && !error && <p className="muted">Reading the workspace…</p>}
          {files && !available.length && (
            <p className="muted">No {allowPdf ? 'text or PDF' : 'text'} files in this workspace. Files must be .tex, .md, .txt{allowPdf ? ', .pdf' : ''} and similar, outside hidden folders.</p>
          )}
          {available.map((file) => (
            <label key={file.path} className="picker-item">
              <input type="checkbox" checked={selected.includes(file.path)} onChange={() => toggle(file.path)} />
              <span className="picker-path">{file.path}</span>
              <span className="muted">{bytes(file.size)}</span>
            </label>
          ))}
        </div>
      )}
    </div>
  )
}

export function Modal({ title, onClose, children, wide }: { title: ReactNode; onClose?: () => void; children: ReactNode; wide?: boolean }) {
  const titleId = useId()
  useEffect(() => {
    if (!onClose) return
    const key = (event: KeyboardEvent) => event.key === 'Escape' && onClose()
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [onClose])
  // A portal: a dialog opened from the mobile drawer must not inherit its transform.
  return createPortal(
    <div className="modal-scrim" onMouseDown={(event) => event.target === event.currentTarget && onClose?.()}>
      <div className={'modal' + (wide ? ' modal-wide' : '')} role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <header className="modal-head">
          <h2 id={titleId}>{title}</h2>
          {onClose && (
            <button type="button" className="icon-btn" aria-label="Close" onClick={onClose}><CloseIcon /></button>
          )}
        </header>
        <div className="modal-body">{children}</div>
      </div>
    </div>,
    document.body,
  )
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return children ? <p className="error-note" role="alert">{children}</p> : null
}

export function Loading({ children = 'Loading…' }: { children?: ReactNode }) {
  return <p className="loading muted">{children}</p>
}

export function SourceLines({ content, highlight, start = 1 }: { content: string; highlight?: [number, number][]; start?: number }) {
  const lines = content.split('\n')
  const read = (n: number) => highlight?.some(([a, b]) => n >= a && n <= b)
  return (
    <pre className="source">
      {lines.map((line, index) => {
        const number = index + start
        return (
          <div key={number} className={'source-line' + (highlight ? (read(number) ? ' source-read' : ' source-unread') : '')}>
            <span className="source-number">{number}</span>
            <span className="source-text">{line || ' '}</span>
          </div>
        )
      })}
    </pre>
  )
}
