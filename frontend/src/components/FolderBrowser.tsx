import { useEffect, useState, type FormEvent } from 'react'
import { api, type FolderListing, type ReadSettings } from '../api'
import { workspaceBusy } from '../concurrency'
import { refreshStatus, useApp } from '../store'
import { ErrorNote, Loading, Modal } from './common'
import { FileIcon } from './Icons'

function counts(files: { tex: number; pdf: number; py: number }) {
  const parts = [files.tex && `${files.tex} TeX`, files.pdf && `${files.pdf} PDF`, files.py && `${files.py} Python`].filter(Boolean)
  return parts.length ? parts.join(', ') : 'No TeX, PDF or Python files'
}

export function FolderBrowser({ onClose }: { onClose: () => void }) {
  const app = useApp()
  const [path, setPath] = useState(app.status?.workspace_path || '')
  const [listing, setListing] = useState<FolderListing | null>(null)
  const [error, setError] = useState('')
  const [name, setName] = useState('')
  const busy = workspaceBusy(app.snapshot)

  useEffect(() => {
    setError('')
    api.folders(path).then(setListing, (reason: Error) => setError(reason.message))
  }, [path])

  const open = async (target: string) => {
    setError('')
    try {
      await api.openFolder(target)
      await refreshStatus()
      onClose()
    } catch (reason) {
      setError((reason as Error).message)
    }
  }
  const create = async (event: FormEvent) => {
    event.preventDefault()
    try {
      const created = await api.createFolder(path, name)
      setName('')
      setPath(created.path)
    } catch (reason) {
      setError((reason as Error).message)
    }
  }
  const rootName = (listing?.root || app.status?.root || '').split('/').filter(Boolean).pop() || 'root'
  const crumbs = path ? path.split('/') : []
  const isCurrent = listing && listing.path === (app.status?.workspace_path || '')
  return (
    <Modal title="Open a folder" onClose={onClose} wide>
      <nav className="crumbs" aria-label="Folder path">
        <button type="button" className="crumb" onClick={() => setPath('')}>{rootName}</button>
        {crumbs.map((part, index) => (
          <span key={index}>
            <span className="crumb-sep">/</span>
            <button type="button" className="crumb" onClick={() => setPath(crumbs.slice(0, index + 1).join('/'))}>{part}</button>
          </span>
        ))}
      </nav>
      <ErrorNote>{error}</ErrorNote>
      {!listing && !error && <Loading />}
      {listing && (
        <>
          <div className="folder-here">
            <span className="muted">{counts(listing.files)} in this folder</span>
            <button type="button" className="btn btn-primary" disabled={!!isCurrent || busy} onClick={() => open(listing.path)}>
              {isCurrent ? 'This is the open folder' : 'Open this folder'}
            </button>
          </div>
          {busy && <p className="busy-note">Pause all active work and cancel queued turns before opening another folder. Chats and jobs belong to the folder they started in.</p>}
          <ul className="folder-list">
            {listing.folders.map((folder) => (
              <li key={folder.path}>
                <button type="button" className="folder-row" onClick={() => setPath(folder.path)}>
                  <span className="folder-glyph" aria-hidden="true" />
                  <span className="folder-name">{folder.name}</span>
                  <span className="folder-count">{counts(folder.files)}</span>
                  {folder.jobs && <span className="pill">Saved work</span>}
                </button>
              </li>
            ))}
            {!listing.folders.length && <li className="muted small folder-empty">No subfolders here.</li>}
          </ul>
          <form className="folder-new" onSubmit={create}>
            <input value={name} onChange={(event) => setName(event.target.value)} placeholder="New folder name" aria-label="New folder name" />
            <button type="submit" className="btn" disabled={!name.trim()}>Create folder here</button>
          </form>
          {listing.recent.length > 0 && (
            <div className="recent">
              <p className="detail-label">Recently opened</p>
              {listing.recent.map((item) => (
                <button key={item || '.'} type="button" className="link-btn recent-item" onClick={() => setPath(item)}>{item || rootName}</button>
              ))}
            </div>
          )}
          <p className="muted small">Only folders inside {listing.root} can be opened. Hidden folders are not shown.</p>
        </>
      )}
    </Modal>
  )
}

const GROUPS: { key: keyof ReadSettings; label: string; hint: string }[] = [
  { key: 'tex', label: 'LaTeX', hint: '.tex .bib .sty .cls' },
  { key: 'pdf', label: 'PDF', hint: 'text extracted locally' },
  { key: 'py', label: 'Python', hint: '.py' },
  { key: 'text', label: 'Other text', hint: '.md .txt .json .yaml .csv' },
]

export function FileAccess({ onClose }: { onClose: () => void }) {
  const { status } = useApp()
  const [error, setError] = useState('')
  if (!status) return null
  const toggle = async (key: keyof ReadSettings, value: boolean) => {
    setError('')
    try {
      await api.saveSettings({ [key]: value })
      await refreshStatus()
    } catch (reason) {
      setError((reason as Error).message)
    }
  }
  return (
    <Modal title="Files the model may read" onClose={onClose}>
      <p>In this folder, the model can list, search and read these kinds of files by itself. Files you pin or drop into a job are always available to that job.</p>
      <div className="access-list">
        {GROUPS.map((group) => (
          <label key={group.key} className="access-item">
            <input type="checkbox" checked={status.read[group.key]} onChange={(event) => toggle(group.key, event.target.checked)} />
            <FileIcon size={16} />
            <span className="access-label">{group.label}</span>
            <span className="muted small">{group.hint}</span>
          </label>
        ))}
      </div>
      <ErrorNote>{error}</ErrorNote>
      <p className="muted small">Saved for this folder and applied to new work. Nothing leaves your computer when the model reads a file.</p>
    </Modal>
  )
}
