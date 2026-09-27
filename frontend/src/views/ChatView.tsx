import { useEffect, useLayoutEffect, useRef, useState, type FormEvent, type KeyboardEvent, type ReactElement } from 'react'
import { api, type LiveTurn, type RouteMode, type TranscriptItem } from '../api'
import { Collapse, ErrorNote, Loading, Toggle, useLoad } from '../components/common'
import { CloseIcon, FileIcon, PauseIcon, SendIcon } from '../components/Icons'
import { EffortSlider, effortLimits, type EffortKind, type EffortLevel } from '../components/Effort'
import { Inline, Markdown, StreamingMarkdown } from '../components/Markdown'
import { proofLook, researchLook, Square, type Variant } from '../components/Square'
import { useDropTarget } from '../drop'
import { ago, count } from '../format'
import { MODES } from '../modes'
import { go, href } from '../router'
import { useApp, useTick } from '../store'
import { BusyNote, Thinking, toolLine } from './shared'

type ChatMode = 'critic' | 'explore' | 'free'

const INTRO: Record<ChatMode, { title: string; text: string; examples: string[] }> = {
  free: {
    title: 'What are you working on?',
    text: 'Describe the task in your own words. The model suggests a workflow (a proof, a critique, a literature or referee report, a write-up) and rewrites your request; nothing runs until you press Start. One conversation can hold several jobs.',
    examples: [
      'Prove that every bounded sequence in $H^1(0,1)$ has a subsequence converging strongly in $L^2(0,1)$.',
      'Referee manuscript.tex and check the step from weak to strong convergence.',
      'Turn my notes in notes.md into a clean LaTeX section using my macros.',
    ],
  },
  critic: {
    title: 'Critique an argument',
    text: 'Paste a statement with its proof, or name a file in the workspace. The critic looks for the first false or unjustified step and tries counterexamples before suggesting repairs.',
    examples: [
      'Is every bounded sequence in $L^2(0,1)$ strongly precompact? Justify your answer.',
      'Read manuscript.tex and identify the first unsupported implication.',
    ],
  },
  explore: {
    title: 'Explore a question',
    text: 'Ask about approaches, reformulations or connections. The model labels heuristic steps and conjectures, so you can tell them from arguments.',
    examples: [
      'Which compactness arguments show that a bounded sequence in $H^1(0,1)$ has a subsequence converging in $L^2(0,1)$?',
      'How could one approach an observability inequality for the heat equation on a bounded domain?',
    ],
  },
}

const ROUTE_LABELS: Record<RouteMode, { noun: string; label: string }> = {
  prove: { noun: 'a proof', label: 'Proof' },
  critic: { noun: 'a critique', label: 'Critique' },
  explore: { noun: 'an exploration', label: 'Exploration' },
  literature: { noun: 'a literature report', label: 'Literature report' },
  referee: { noun: 'a referee report', label: 'Referee report' },
  writeup: { noun: 'a write-up', label: 'Write-up' },
}
const JOB_ROUTES: RouteMode[] = ['prove', 'literature', 'referee', 'writeup']

export function ChatView({ mode, id }: { mode: ChatMode; id: string | null }) {
  const app = useApp()
  const tick = useTick(id)
  const { data, error } = useLoad(() => (id ? api.chat(id) : Promise.resolve(null)), [id, tick])
  const task = app.snapshot.task
  const busy = !!task && ['starting', 'running', 'pausing'].includes(task.state)
  const ours = busy && task!.kind === 'chat' && task!.target === id
  const current = ours && app.snapshot.live && app.snapshot.live.chat === id ? app.snapshot.live : null
  // Keep a finished turn on screen until the saved transcript replaces it.
  const [lingering, setLingering] = useState<LiveTurn | null>(null)
  const previous = useRef<LiveTurn | null>(null)
  useEffect(() => {
    if (!current && previous.current) setLingering(previous.current)
    previous.current = current
  }, [current])
  useEffect(() => { setLingering(null) }, [data])
  const live = current || lingering
  const routing = !!id && app.snapshot.routing.includes(id)
  const [draft, setDraft] = useState('')
  const [attachments, setAttachments] = useState<string[]>([])
  const [sendError, setSendError] = useState('')
  const [sending, setSending] = useState(false)
  const [think, setThink] = useState<boolean>(app.status?.defaults.think ?? true)
  const [online, setOnline] = useState<boolean>(!!app.status?.online)
  const locked = !!app.status?.online_locked
  const scroller = useRef<HTMLDivElement>(null)
  const pinned = useRef(true)

  useDropTarget('Attached to your next message', (paths) => setAttachments((items) => [...new Set([...items, ...paths])]))
  useEffect(() => { if (data) { setThink(data.settings.think); setOnline(!!data.settings.online) } }, [data])

  // Follow new output while the reader is at the bottom; never yank them back up.
  const liveSize = live ? live.steps.reduce((n, step) => n + (step.type === 'call' ? step.text.length + step.thinking.length : 1), 0) : 0
  useLayoutEffect(() => {
    const element = scroller.current
    if (element && pinned.current) element.scrollTop = element.scrollHeight
  }, [data, liveSize, routing])

  const free = mode === 'free'
  const send = async (text?: string) => {
    const content = (text ?? draft).trim()
    if (!content || sending) return
    setSending(true)
    setSendError('')
    try {
      let chatId = id
      if (!chatId) chatId = (await api.createChat(mode, think, locked ? undefined : online)).id
      if (free) await api.route(chatId, content, attachments)
      else await api.send(chatId, content, attachments)
      setDraft('')
      setAttachments([])
      pinned.current = true
      if (!id) go(mode, chatId)
    } catch (reason) {
      setSendError((reason as Error).message)
    } finally {
      setSending(false)
    }
  }
  const submit = (event: FormEvent) => { event.preventDefault(); send() }
  const keys = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); send() }
  }
  const changeThink = (value: boolean) => {
    setThink(value)
    if (id) api.chatSettings(id, { think: value }).catch((reason: Error) => setSendError(reason.message))
  }
  const changeOnline = (value: boolean) => {
    setOnline(value)
    if (id) api.chatSettings(id, { online: value }).catch((reason: Error) => { setOnline(!value); setSendError(reason.message) })
  }
  const transcript = data?.transcript || []
  const last = [...transcript].reverse().find((item) => item.role !== 'notice' && item.role !== 'route')
  const canReview = !!id && !busy && !!last && last.role === 'assistant' && !!last.content
  // Free mode may read a new message while a job runs; jobs it starts wait in the queue.
  const blocked = free ? routing : busy

  return (
    <div className="chat">
      <div className="chat-scroll sheet paper" ref={scroller}
        onScroll={(event) => { const el = event.currentTarget; pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80 }}>
        <div className="sheet-inner chat-inner">
          {!id && <Intro mode={mode} onPick={(text) => setDraft(text)} />}
          {id && error && <ErrorNote>{error}</ErrorNote>}
          {id && !data && !error && <Loading />}
          {data && !transcript.length && !live && !routing && <Intro mode={mode} onPick={(text) => setDraft(text)} />}
          {id && <Transcript items={transcript} mode={mode} chatId={id} />}
          {routing && (
            <div className="entry message message-model">
              <span className="in-margin speaker speaker-model mode-free">∀</span>
              <p className="muted routing"><Square variant="running" size={12} /> Reading your request to suggest a workflow…</p>
            </div>
          )}
          {live && <LiveTurnView live={live} mode={mode} />}
        </div>
      </div>
      <form className="composer" onSubmit={submit}>
        {attachments.length > 0 && (
          <div className="attachments composer-attachments">
            {attachments.map((path) => (
              <span key={path} className="chip">
                <FileIcon size={14} /><span>{path}</span>
                <button type="button" className="chip-remove" aria-label={`Remove ${path}`}
                  onClick={() => setAttachments((items) => items.filter((item) => item !== path))}><CloseIcon size={13} /></button>
              </span>
            ))}
          </div>
        )}
        <div className="composer-box">
          <textarea value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={keys}
            rows={Math.min(8, Math.max(2, draft.split('\n').length))}
            placeholder={free ? 'Describe what you want to do; drop files here to attach them…'
              : mode === 'critic' ? 'Ask the critic to check a statement, a proof, or a file…' : 'Ask about an approach, a reformulation or a connection…'}
            aria-label="Message" />
          {ours ? (
            <button type="button" className="btn composer-send" disabled={task?.state === 'pausing'} onClick={() => api.pause().catch((reason: Error) => setSendError(reason.message))}>
              <PauseIcon size={16} /> Stop
            </button>
          ) : (
            <button type="submit" className="btn btn-primary composer-send" disabled={blocked || sending || !draft.trim()} aria-label="Send">
              <SendIcon size={16} /> Send
            </button>
          )}
        </div>
        <div className="composer-bar">
          <Toggle label="Thinking" checked={think} onChange={changeThink} />
          <span title={locked ? 'Launched with --offline: online search stays off' : online ? 'Search queries leave this computer' : 'Local and cached sources only'}>
            <Toggle label="Search online" checked={online && !locked} onChange={changeOnline} disabled={locked} />
          </span>
          {canReview && (
            <button type="button" className="btn btn-quiet btn-small" onClick={() => api.review(id!).catch((reason: Error) => setSendError(reason.message))}>
              Fresh review of the last answer
            </button>
          )}
          <span className="composer-hint">Enter sends, Shift+Enter adds a line, drop files to attach</span>
        </div>
        {busy && !ours && !free && <BusyNote />}
        <ErrorNote>{sendError}</ErrorNote>
      </form>
    </div>
  )
}

function Intro({ mode, onPick }: { mode: ChatMode; onPick: (text: string) => void }) {
  const intro = INTRO[mode]
  return (
    <div className="entry intro">
      <span className={`in-margin speaker speaker-model mode-${mode}`}>{MODES[mode].glyph}</span>
      <div>
        <h1 className="page-title">{intro.title}</h1>
        <p className="lede">{intro.text}</p>
        <div className="examples">
          {intro.examples.map((example) => (
            <button key={example} type="button" className="example" onClick={() => onPick(example)}>
              <Markdown>{example}</Markdown>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

type Step = { item: TranscriptItem; index: number }
type Turn = { user?: TranscriptItem; steps: Step[] }

function group(items: TranscriptItem[]): Turn[] {
  const turns: Turn[] = []
  items.forEach((item, index) => {
    if (item.role === 'user' || !turns.length) turns.push({ user: item.role === 'user' ? item : undefined, steps: item.role === 'user' ? [] : [{ item, index }] })
    else turns[turns.length - 1].steps.push({ item, index })
  })
  return turns
}

function Transcript({ items, mode, chatId }: { items: TranscriptItem[]; mode: ChatMode; chatId: string }) {
  return (
    <>
      {group(items).map((turn, index) => (
        <div key={index} className="turn">
          {turn.user && <UserMessage item={turn.user} />}
          <ModelSteps steps={turn.steps} mode={mode} chatId={chatId} />
        </div>
      ))}
    </>
  )
}

function UserMessage({ item }: { item: TranscriptItem }) {
  return (
    <div className={'entry message message-user' + (item.discarded ? ' message-discarded' : '')}>
      <span className="in-margin speaker speaker-user">You</span>
      <div className="message-body">
        <Markdown>{item.content}</Markdown>
        {item.files && item.files.length > 0 && (
          <div className="attachments">{item.files.map((path) => <span key={path} className="chip chip-static"><FileIcon size={13} /><span>{path}</span></span>)}</div>
        )}
        {item.time && <span className="message-time">{ago(item.time)}</span>}
      </div>
    </div>
  )
}

function ModelSteps({ steps, mode, chatId }: { steps: Step[]; mode: ChatMode; chatId: string }) {
  const out: ReactElement[] = []
  let blocks: ReactElement[] = []
  // Free mode mixes workflows: draw each answer with the one that produced it.
  const fallback = mode === 'free' ? 'critic' : mode
  let glyph: 'critic' | 'explore' = fallback
  let routed: 'critic' | 'explore' | null = null  // older answers: take the card above them
  const flush = (key: number) => {
    if (!blocks.length) return
    out.push(
      <div key={'answer' + key} className="entry message message-model">
        <span className={`in-margin speaker speaker-model mode-${glyph}`} title={MODES[glyph].label}>{MODES[glyph].glyph}</span>
        <div className="message-body">{blocks}</div>
      </div>,
    )
    blocks = []
  }
  for (let i = 0; i < steps.length; i++) {
    const { item, index } = steps[i]
    if (item.role === 'route') routed = item.mode === 'critic' || item.mode === 'explore' ? item.mode : null
    const answeredBy = item.mode === 'critic' || item.mode === 'explore' ? item.mode
      : item.role === 'review' ? 'critic' : routed || fallback
    if (item.role !== 'route' && item.role !== 'notice' && answeredBy !== glyph) {
      flush(index)
      glyph = answeredBy
    }
    if (item.role === 'route') {
      flush(index)
      out.push(<RouteCard key={'route' + index} chatId={chatId} index={index} item={item} />)
    } else if (item.role === 'assistant') {
      const calls = item.tool_calls || []
      const results: TranscriptItem[] = []
      while (i + 1 < steps.length && steps[i + 1].item.role === 'tool' && results.length < calls.length) results.push(steps[++i].item)
      const final = !calls.length
      blocks.push(
        <div key={index} className="step">
          <Thinking text={item.thinking || ''} />
          {calls.map((call, n) => (
            <Collapse key={n} className="tool-step" summary={<span className="tool-line">{toolLine(call)}</span>}>
              <pre className="plain">{results[n]?.content ?? 'No result was recorded.'}</pre>
            </Collapse>
          ))}
          {item.content && <Markdown className={final ? 'answer' : 'interim'}>{item.content}</Markdown>}
          {final && item.stats?.eval_count !== undefined && (
            <span className="message-time">{count(item.stats.eval_count)} tokens{item.stats.done_reason === 'length' ? ', stopped at the output limit' : ''}</span>
          )}
        </div>,
      )
    } else if (item.role === 'review') {
      blocks.push(
        <div key={index} className="review-note">
          <p className="review-label">Fresh-context review</p>
          <p className="muted small">A separate critic context audited the previous answer. It is another model opinion, not verification.</p>
          <Thinking text={item.thinking || ''} />
          <Markdown>{item.content}</Markdown>
        </div>,
      )
    } else if (item.role === 'notice') {
      blocks.push(<p key={index} className={'notice' + (item.discarded ? ' notice-discarded' : '')}>{item.content}</p>)
    } else if (item.role === 'tool') {
      blocks.push(<Collapse key={index} className="tool-step" summary={<span className="tool-line">{item.tool_name || 'tool'} result</span>}><pre className="plain">{item.content}</pre></Collapse>)
    }
  }
  flush(-1)
  return <>{out}</>
}

function RouteCard({ chatId, index, item }: { chatId: string; index: number; item: TranscriptItem }) {
  const app = useApp()
  const proposed = (item.mode === 'clarify' || !item.mode ? 'explore' : item.mode) as RouteMode
  const [mode, setMode] = useState<RouteMode>(proposed)
  const [request, setRequest] = useState(item.request || '')
  const [files, setFiles] = useState<string[]>(item.files || [])
  const [error, setError] = useState(item.error || '')
  const [starting, setStarting] = useState(false)
  const editable = ['proposed', 'failed', 'cancelled'].includes(item.status || '')
  const busy = !!app.snapshot.task && ['starting', 'running', 'pausing'].includes(app.snapshot.task.state)
  const [level, setLevel] = useState<EffortLevel>('medium')
  const effortKind: EffortKind | null = mode === 'prove' ? 'proof' : mode === 'writeup' ? 'writeup' : mode === 'literature' || mode === 'referee' ? 'research' : null
  useEffect(() => { setError(item.error || '') }, [item.error])
  const start = async () => {
    setStarting(true)
    setError('')
    try {
      const defaults = app.status?.defaults
      const limits = effortKind && defaults ? effortLimits(effortKind, defaults, level) : undefined
      await api.startRoute(chatId, index, { mode, request, files, limits: limits as Record<string, number> | undefined })
    } catch (reason) {
      setError((reason as Error).message)
    } finally {
      setStarting(false)
    }
  }
  if (item.status === 'dismissed') {
    return <div className="entry route-dismissed"><span className="in-margin" /><span className="muted small">Suggestion dismissed.</span></div>
  }
  const shown = editable ? mode : proposed
  return (
    <div className={'entry route route-' + (item.status || 'proposed')}>
      <span className={`in-margin speaker speaker-model mode-${shown}`} title={ROUTE_LABELS[shown].label}>{MODES[shown].glyph}</span>
      <div className="route-card">
        {editable && item.mode === 'clarify' && (
          <>
            <p className="route-title">One detail first</p>
            <Markdown className="route-question">{item.question || ''}</Markdown>
            <p className="muted small">Answer in the message box, or choose a workflow yourself below.</p>
          </>
        )}
        {editable && item.mode !== 'clarify' && (
          <>
            <p className="route-title">Looks like {ROUTE_LABELS[proposed].noun}</p>
            {item.reason && <p className="muted small route-reason">{item.reason}</p>}
          </>
        )}
        {editable ? (
          <>
            <div className="route-modes" role="radiogroup" aria-label="Workflow">
              {(Object.keys(ROUTE_LABELS) as RouteMode[]).map((key) => (
                <button key={key} type="button" role="radio" aria-checked={mode === key}
                  className={'route-mode' + (mode === key ? ' route-mode-active' : '')} onClick={() => setMode(key)}>
                  <span className={`route-cover mode-${key}`}>{MODES[key].glyph}</span>{ROUTE_LABELS[key].label}
                </button>
              ))}
            </div>
            <label className="field">
              <span className="field-label">Request, as the {ROUTE_LABELS[mode].label.toLowerCase()} will see it</span>
              <textarea className="route-request" rows={Math.min(10, Math.max(3, request.split('\n').length + 1))} value={request}
                onChange={(event) => setRequest(event.target.value)} />
              <span className="field-hint">{JOB_ROUTES.includes(mode)
                ? 'Jobs do not see this conversation, so the request must contain every hypothesis they need.'
                : 'Answered in this conversation, with its earlier turns as context.'}</span>
            </label>
            {files.length > 0 && (
              <div className="attachments">
                {files.map((path) => (
                  <span key={path} className="chip"><FileIcon size={14} /><span>{path}</span>
                    <button type="button" className="chip-remove" aria-label={`Remove ${path}`} onClick={() => setFiles(files.filter((f) => f !== path))}><CloseIcon size={13} /></button>
                  </span>
                ))}
              </div>
            )}
            {item.missing && item.missing.length > 0 && <p className="muted small">Not found in this folder: {item.missing.join(', ')}.</p>}
            {effortKind && app.status && (
              <EffortSlider kind={effortKind} defaults={app.status.defaults} level={level} onLevel={setLevel} compact />
            )}
            {(mode === 'literature' || mode === 'referee') && (
              <p className="muted small">Online search follows this conversation's switch below the message box.</p>
            )}
            <div className="route-actions">
              <button type="button" className="btn btn-primary" disabled={starting || !request.trim()} onClick={start}>
                {starting ? 'Starting…' : `Start ${ROUTE_LABELS[mode].label.toLowerCase()}`}
              </button>
              <button type="button" className="btn btn-quiet" onClick={() => api.dismissRoute(chatId, index).catch((reason: Error) => setError(reason.message))}>Dismiss</button>
              {busy && JOB_ROUTES.includes(mode) && <span className="muted small">The model is busy; this job will wait its turn.</span>}
              {busy && !JOB_ROUTES.includes(mode) && <span className="muted small">Pause the running job first: answers need the model now.</span>}
            </div>
          </>
        ) : (
          <StartedRoute item={item} />
        )}
        <ErrorNote>{error}</ErrorNote>
      </div>
    </div>
  )
}

function StartedRoute({ item }: { item: TranscriptItem }) {
  const app = useApp()
  const mode = item.mode as RouteMode
  const label = ROUTE_LABELS[mode]?.label || 'Job'
  let status = item.status === 'queued' ? 'Waiting for the model' : item.status === 'starting' ? 'Starting' : ''
  let variant: Variant = item.status === 'queued' ? 'ready' : 'running'
  if (item.job_id && mode === 'prove') {
    const job = app.proofs?.find((proof) => proof.id === item.job_id)
    if (job) ({ variant, label: status } = proofLook(job.status, job.running))
  } else if (item.job_id) {
    const job = app.research?.find((report) => report.id === item.job_id)
    if (job) ({ variant, label: status } = researchLook(job.status, job.running))
  } else if (!JOB_ROUTES.includes(mode) && item.status === 'started') {
    status = 'Answered below'
    variant = 'complete'
  }
  return (
    <div className="route-started">
      <p className="route-title"><Square variant={variant} size={14} /> {label}{status && <span className="route-status">{status}</span>}</p>
      <div className="route-summary"><Inline limit={260}>{item.request || ''}</Inline></div>
      {item.job_id && <a className="btn btn-small" href={href(mode, item.job_id)}>Open the {label.toLowerCase()}</a>}
    </div>
  )
}

function LiveTurnView({ live, mode }: { live: LiveTurn; mode: ChatMode }) {
  const glyph = live.mode || (mode === 'free' ? 'critic' : mode)
  return (
    <div className="turn turn-live">
      {live.kind === 'message' && live.user && <UserMessage item={{ role: 'user', content: live.user, time: '' }} />}
      <div className="entry message message-model">
        <span className={`in-margin speaker speaker-model mode-${glyph}`}>{MODES[glyph].glyph}</span>
        <div className="message-body">
          {live.kind === 'review' && <p className="review-label">Fresh-context review in progress</p>}
          {live.steps.map((step, index) => step.type === 'call' ? (
            <div key={index} className="step">
              <Thinking text={step.thinking} live={!step.text} />
              {step.text && <StreamingMarkdown text={step.text} className="answer" />}
              {!step.text && !step.thinking && index === live.steps.length - 1 && (
                <p className="muted small">Waiting for the first tokens. The model may still be loading into memory.</p>
              )}
            </div>
          ) : step.type === 'tool' ? (
            <p key={index} className="tool-line tool-live">{step.text}</p>
          ) : step.type === 'result' ? (
            <p key={index} className="tool-result-live">{step.text}</p>
          ) : (
            <p key={index} className="notice">{step.text}</p>
          ))}
          {!live.steps.length && <p className="muted small">Waiting for the model…</p>}
        </div>
      </div>
    </div>
  )
}
