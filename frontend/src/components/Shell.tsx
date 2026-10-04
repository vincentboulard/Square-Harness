import { useEffect, useRef, useState } from 'react'
import { api, type ChatListItem, type TaskSummary } from '../api'
import { chatDeleteBlocked } from '../chatDeletion'
import { ago, duration } from '../format'
import { MODES } from '../modes'
import { setPanelHidden } from '../panels'
import { href, notebook, RAIL, type Mode, type Route } from '../router'
import { applyTheme, deleteConversation, readTheme, useApp, type AppState, type Theme } from '../store'
import { ErrorNote, Modal } from './common'
import { FileAccess, FolderBrowser } from './FolderBrowser'
import { Inline } from './Markdown'
import { Glyph } from './Glyph'
import { AutoIcon, ChevronIcon, CloseIcon, FileIcon, MenuIcon, MoonIcon, PanelLeftIcon, PauseIcon, PhoneIcon, PlusIcon, SunIcon, TrashIcon } from './Icons'
import { proofLook, researchLook, reviewStatusLook, Square } from './Square'

export function BrandMark({ size = 32 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true" className="brand-mark">
      <rect width="64" height="64" rx="12" className="brand-cover" />
      <path d="M0 16h64M0 32h64M0 48h64M16 0v64M32 0v64M48 0v64" className="brand-grid" />
      <path d="M13 0v64" className="brand-margin" />
      <rect x="33" y="33" width="19" height="19" rx="1.5" className="brand-square" />
    </svg>
  )
}

/** Which mode the running task belongs to, so its notebook shows activity. */
export function taskMode(task: TaskSummary | null, app: AppState): Mode | null {
  if (!task || !['starting', 'running', 'pausing'].includes(task.state)) return null
  if (task.kind === 'proof') return 'prove'
  if (task.kind === 'research') {
    const job = app.research?.find((item) => item.id === task.target)
    return job ? job.kind : task.label.startsWith('Review') ? 'referee' : task.label.startsWith('Write') ? 'writeup' : 'literature'
  }
  const chat = app.chats?.find((item) => item.id === task.target)
  return chat ? chat.kind : null
}

export function taskHref(task: TaskSummary, app: AppState) {
  const mode = taskMode(task, app) || (task.kind === 'proof' ? 'prove' : 'free')
  return href(mode, task.target)
}

export function Rail({ route }: { route: Route }) {
  const app = useApp()
  const running = taskMode(app.snapshot.task, app)
  const busy = running && notebook(running)
  const about = route.page === 'about'
  return (
    <nav className="rail" aria-label="Modes">
      <a className={'rail-brand' + (about ? ' rail-brand-active' : '')} href="#/about" title="About Square Harness"
        aria-label="About Square Harness" aria-current={about ? 'page' : undefined}><BrandMark size={40} /></a>
      {RAIL.map((mode) => {
        const info = MODES[mode]
        const active = !about && notebook(route.mode) === mode
        return (
          <a key={mode} href={href(mode)} className={'mode' + (active ? ' mode-active' : '')}
            aria-current={active ? 'page' : undefined} title={info.summary}>
            <span className={`mode-cover mode-${mode}`}>
              <span className="mode-glyph"><Glyph mode={mode} /></span>
              {busy === mode && <span className="mode-busy" aria-label="Running" />}
            </span>
            <span className="mode-label">{info.label}</span>
          </a>
        )
      })}
      <span className="rail-spacer" />
      <ThemeButton />
    </nav>
  )
}

function ThemeButton() {
  const [theme, setTheme] = useState<Theme>(readTheme)
  const next: Record<Theme, Theme> = { system: 'light', light: 'dark', dark: 'system' }
  const label = { system: 'Theme follows the system', light: 'Light theme', dark: 'Dark theme' }[theme]
  return (
    <button type="button" className="rail-button" title={label + ' (click to change)'} aria-label={label}
      onClick={() => { const value = next[theme]; setTheme(value); applyTheme(value) }}>
      {theme === 'light' ? <SunIcon /> : theme === 'dark' ? <MoonIcon /> : <AutoIcon />}
    </button>
  )
}

function ReviewTally({ reviews }: { reviews: Partial<Record<string, number>> }) {
  const order = ['no_issue_found', 'issues_found', 'uncertain', 'review_unavailable']
  const shown = order.filter((status) => reviews[status])
  if (!shown.length) return null
  return (
    <span className="tally">
      {shown.map((status) => (
        <span key={status} className="tally-item" title={`${reviews[status]} ${reviewStatusLook(status).label.toLowerCase()}`}>
          <Square variant={reviewStatusLook(status).variant} size={11} />{reviews[status]}
        </span>
      ))}
    </span>
  )
}

export function Sidebar({ route }: { route: Route }) {
  const app = useApp()
  const info = MODES[notebook(route.mode)]
  return (
    <aside className="sidebar" aria-label={info.label}>
      <div className="sidebar-inner">
        <header className="side-head">
          <div className="side-title-row">
            <h1 className="side-title">{info.label}</h1>
            <button type="button" className="icon-btn side-hide" title="Hide this list" aria-label="Hide this list"
              onClick={() => setPanelHidden('sidebar', true)}>
              <PanelLeftIcon />
            </button>
          </div>
          <p className="side-summary">{info.summary}</p>
          <a className="btn btn-primary btn-block" href={href(info.mode)}><PlusIcon size={16} /> {info.newLabel}</a>
        </header>
        <div className="side-list">
          <SideList route={route} app={app} />
        </div>
        <TaskPanel />
        <WorkspacePanel />
      </div>
    </aside>
  )
}

/** What is left of the list once it slides away: the notebook's spine, to pull it back. */
export function SideSpine({ route }: { route: Route }) {
  const app = useApp()
  const info = MODES[notebook(route.mode)]
  const running = !!app.snapshot.task && ['starting', 'running', 'pausing'].includes(app.snapshot.task.state)
  return (
    <button type="button" className="spine side-spine" onClick={() => setPanelHidden('sidebar', false)}
      title={`Show the ${info.label} list`} aria-label={`Show the ${info.label} list`}>
      <ChevronIcon size={16} />
      <span className="spine-text">{info.label}</span>
      {running && <Square variant="running" size={12} label="The model is working" />}
    </button>
  )
}

function SideList({ route, app }: { route: Route; app: AppState }) {
  const info = MODES[notebook(route.mode)]
  if (info.kind === 'proof') {
    if (!app.proofs) return <p className="side-empty muted">Loading…</p>
    if (!app.proofs.length) return <p className="side-empty muted">{info.empty}</p>
    return (
      <>
        {app.proofs.map((job) => {
          const look = proofLook(job.status, job.running, job.version)
          return (
            <a key={job.id} href={href('prove', job.id)} className={'side-item' + (route.id === job.id ? ' side-item-active' : '')}>
              <Square variant={look.variant} label={look.label} />
              <span className="side-body">
                <span className="side-name"><Inline limit={110}>{job.title}</Inline></span>
                <span className="side-meta">
                  {job.version === 2
                    ? <span>Attempt {job.rounds.used} of {job.rounds.limit}</span>
                    : <span title="Made by the v0.4 engine; readable, not resumable">v0.4 job</span>}
                  <ReviewTally reviews={job.reviews} />
                  <span className="side-time">{ago(job.updated_at)}</span>
                </span>
              </span>
            </a>
          )
        })}
      </>
    )
  }
  if (info.kind === 'chat' || info.kind === 'free') {
    return <ChatSideList route={route} app={app} />
  }
  if (!app.research) return <p className="side-empty muted">Loading…</p>
  const jobs = app.research.filter((job) => job.kind === info.mode)
  if (!jobs.length) return <p className="side-empty muted">{info.empty}</p>
  return (
    <>
      {jobs.map((job) => {
        const look = researchLook(job.status, job.running)
        return (
          <a key={job.id} href={href(info.mode, job.id)} className={'side-item' + (route.id === job.id ? ' side-item-active' : '')}>
            <Square variant={look.variant} label={look.label} />
            <span className="side-body">
              <span className="side-name"><Inline limit={110}>{job.title}</Inline></span>
              <span className="side-meta">
                <span>{look.label}</span>
                <span className="side-time">{ago(job.updated_at)}</span>
              </span>
            </span>
          </a>
        )
      })}
    </>
  )
}

function ChatSideList({ route, app }: { route: Route; app: AppState }) {
  const [selected, setSelected] = useState<ChatListItem | null>(null)
  const opener = useRef<HTMLButtonElement | null>(null)
  useEffect(() => {
    if (selected && app.chats && !app.chats.some((chat) => chat.id === selected.id)) setSelected(null)
  }, [app.chats, selected])
  // Every conversation lives here, including critiques and explorations from earlier versions.
  if (!app.chats) return <p className="side-empty muted">Loading…</p>
  return (
    <>
      {!app.chats.length && <p className="side-empty muted">{MODES.free.empty}</p>}
      {app.chats.map((chat) => {
        const blocked = chatDeleteBlocked(app.snapshot, chat.id)
        return (
          <div key={chat.id} className={'side-chat-row' + (route.id === chat.id ? ' side-item-active' : '')}>
            <a href={href(chat.kind, chat.id)} className="side-item side-item-chat" aria-current={route.id === chat.id ? 'page' : undefined}>
              <span className="side-body">
                <span className="side-name"><Inline limit={110}>{chat.title}</Inline></span>
                <span className="side-meta">
                  {chat.kind !== 'free' && <span className="side-kind" title="A conversation from an earlier version">{MODES[chat.kind].label}</span>}
                  <span className="side-preview">{chat.preview ? <Inline limit={80}>{chat.preview}</Inline> : 'No answer yet'}</span>
                  <span className="side-time">{ago(chat.updated_at)}</span>
                </span>
              </span>
            </a>
            <button type="button" className="icon-btn side-delete" disabled={blocked}
              aria-label={`Delete conversation: ${chat.title}${blocked ? '. Pause or cancel its work first.' : ''}`}
              title={blocked ? 'Pause this conversation or cancel its queued work before deleting it.' : 'Delete conversation'}
              onClick={(event) => { opener.current = event.currentTarget; setSelected(chat) }}>
              <TrashIcon size={17} />
            </button>
          </div>
        )
      })}
      {selected && <DeleteChatDialog key={selected.id} chat={selected} app={app} opener={opener.current} onClose={() => setSelected(null)} />}
    </>
  )
}

function DeleteChatDialog({ chat, app, opener, onClose }: { chat: ChatListItem; app: AppState; opener: HTMLButtonElement | null; onClose: () => void }) {
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState('')
  const body = useRef<HTMLDivElement>(null)
  const cancel = useRef<HTMLButtonElement>(null)
  const blocked = chatDeleteBlocked(app.snapshot, chat.id)
  useEffect(() => {
    if (deleting) body.current?.closest<HTMLElement>('[role="dialog"]')?.focus()
  }, [deleting])
  useEffect(() => {
    const dialog = body.current?.closest<HTMLElement>('[role="dialog"]')
    if (!dialog) return
    dialog.tabIndex = -1
    cancel.current?.focus()
    const trap = (event: KeyboardEvent) => {
      if (event.key !== 'Tab') return
      const buttons = [...dialog.querySelectorAll<HTMLButtonElement>('button:not(:disabled)')]
      const first = buttons[0]
      const last = buttons[buttons.length - 1]
      if (!first) { event.preventDefault(); dialog.focus() }
      else if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog)) {
        event.preventDefault(); last.focus()
      } else if (!event.shiftKey && (document.activeElement === last || document.activeElement === dialog)) {
        event.preventDefault(); first.focus()
      }
    }
    dialog.addEventListener('keydown', trap)
    return () => { dialog.removeEventListener('keydown', trap); if (opener?.isConnected) opener.focus() }
  }, [opener])
  async function remove() {
    if (blocked || deleting) return
    setDeleting(true)
    setError('')
    try { await deleteConversation(chat.id); onClose() }
    catch (reason) { setError((reason as Error).message); setDeleting(false) }
  }
  return (
    <Modal title="Delete conversation?" onClose={deleting ? undefined : onClose}>
      <div ref={body}>
        <p className="delete-chat-title"><strong><Inline>{chat.title}</Inline></strong></p>
        <p>The conversation and its messages will be permanently deleted. Saved proofs, reports and files will remain.</p>
        {blocked && <p className="muted">Pause this conversation or cancel its queued work before deleting it.</p>}
        <ErrorNote>{error}</ErrorNote>
        <div className="modal-actions">
          <button ref={cancel} type="button" className="btn" disabled={deleting} onClick={onClose}>Cancel</button>
          <button type="button" className="btn btn-delete" disabled={blocked || deleting} onClick={remove}>
            <TrashIcon size={16} /> {deleting ? 'Deleting…' : 'Delete conversation'}
          </button>
        </div>
      </div>
    </Modal>
  )
}

export function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [])
  return <>{duration(Math.max(0, now / 1000 - since))}</>
}

export function TaskPanel() {
  const app = useApp()
  const task = app.snapshot.task
  const [error, setError] = useState('')
  if (!task || !['starting', 'running', 'pausing'].includes(task.state)) return null
  const pausing = task.state === 'pausing'
  return (
    <>
    <div className="task-panel" role="status">
      <Square variant="running" size={16} label="Running" />
      <a className="task-text" href={taskHref(task, app)}>
        <span className="task-label">{task.label}</span>
        <span className="task-title"><Inline limit={70}>{task.title}</Inline></span>
        <span className="task-state">{pausing ? 'Pausing at the next checkpoint…' : <>Using the model for <Elapsed since={task.started} /></>}</span>
        {error && <span className="task-error">{error}</span>}
      </a>
      <button type="button" className="btn btn-small" disabled={pausing}
        onClick={() => api.pause().catch((reason: Error) => setError(reason.message))}>
        <PauseIcon size={15} /> Pause
      </button>
    </div>
    <QueuePanel />
    </>
  )
}

function QueuePanel() {
  const app = useApp()
  const [error, setError] = useState('')
  if (!app.snapshot.queue.length) return null
  return (
    <div className="queue-panel">
      <p className="detail-label">Waiting for the model</p>
      {app.snapshot.queue.map((item) => (
        <div key={item.id} className="queue-item">
          <span className="queue-text"><strong>{item.label}</strong> <Inline limit={50}>{item.title}</Inline></span>
          <button type="button" className="icon-btn" aria-label="Cancel this job" title="Cancel"
            onClick={() => api.cancelQueued(item.id).catch((reason: Error) => setError(reason.message))}>
            <CloseIcon size={14} />
          </button>
        </div>
      ))}
      {error && <p className="task-error">{error}</p>}
    </div>
  )
}

function WorkspacePanel() {
  const { status } = useApp()
  const [pairing, setPairing] = useState(false)
  const [browsing, setBrowsing] = useState(false)
  const [access, setAccess] = useState(false)
  if (!status) return null
  const readable = (['tex', 'pdf', 'py', 'text'] as const).filter((key) => status.read[key])
  const readLabel = readable.length === 4 ? 'Reads all file types' : readable.length ? 'Reads ' + readable.join(', ') : 'Reads no files by itself'
  const folder = status.workspace.split('/').filter(Boolean).pop() || status.workspace
  const reachable = status.ollama.reachable
  const installed = status.ollama.models.includes(status.model)
  const ollama = status.backend === 'ollama'
  const modelNote = !reachable ? (ollama ? 'Ollama is not reachable' : 'The model server is not reachable')
    : !installed ? (ollama ? 'Model is not installed' : 'Model is not served') : 'Model ready'
  const modelHint = !reachable ? (ollama ? 'Start ollama serve, then reload.' : 'Start the model server, then reload.')
    : ollama ? `Run ollama pull ${status.model}.` : `Serve ${status.model}, or restart with --model.`
  return (
    <footer className="workspace">
      <div className="workspace-row">
        <button type="button" className="workspace-name workspace-open" title={status.workspace + ' (open another folder)'} onClick={() => setBrowsing(true)}>
          <span className="folder-glyph" aria-hidden="true" />{folder}
        </button>
        <span className={'pill' + (status.online ? ' pill-online' : '')} title={status.online ? 'Search queries and downloads can leave this computer' : 'No external requests; local and cached sources only'}>
          {status.online ? 'Online' : 'Offline'}
        </span>
      </div>
      <div className="workspace-row">
        <span className={'model-dot' + (reachable && installed ? ' model-ok' : ' model-bad')} aria-hidden="true" />
        <span className="workspace-model" title={`${status.model} at ${status.host}: ${modelNote}`}>{status.model}</span>
        {status.lan && (
          <button type="button" className="icon-btn" title="Connect a phone" aria-label="Connect a phone" onClick={() => setPairing(true)}>
            <PhoneIcon size={16} />
          </button>
        )}
      </div>
      <button type="button" className="workspace-access" onClick={() => setAccess(true)}>
        <FileIcon size={13} /> {readLabel}
      </button>
      {browsing && <FolderBrowser onClose={() => setBrowsing(false)} />}
      {access && <FileAccess onClose={() => setAccess(false)} />}
      {(!reachable || !installed) && <p className="workspace-warning">{modelNote}. {modelHint}</p>}
      {pairing && (
        <Modal title="Connect a phone" onClose={() => setPairing(false)}>
          <p>On a phone connected to the same network, open one of these addresses. It contains this workspace's access token, so share it only with your own devices.</p>
          {(status.pair_urls || []).map((url) => <p key={url}><code className="pair-url">{url}</code></p>)}
          <p className="muted">After the first visit the phone stays paired. On iPhone, use Share, then Add to Home Screen; on Android, use the browser menu's Install or Add to Home screen.</p>
        </Modal>
      )}
    </footer>
  )
}

export function TopBar({ title, onMenu }: { title: string; onMenu: () => void }) {
  const app = useApp()
  const running = app.snapshot.task && ['starting', 'running', 'pausing'].includes(app.snapshot.task.state)
  return (
    <header className="topbar">
      <button type="button" className="icon-btn" aria-label="Open navigation" onClick={onMenu}><MenuIcon /></button>
      <span className="topbar-title">{title}</span>
      {running && <Square variant="running" size={16} label="The model is working" />}
    </header>
  )
}
