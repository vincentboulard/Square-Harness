import { useEffect, useLayoutEffect, useRef, useState, type FormEvent, type KeyboardEvent, type ReactElement } from 'react'
import { api, type LiveTurn, type RouteMode, type TranscriptItem } from '../api'
import { assistantResumable, routeActivity } from '../assistant'
import { Collapse, ErrorNote, Loading, Toggle, useLoad } from '../components/common'
import { CloseIcon, FileIcon, PauseIcon, PlayIcon, SendIcon } from '../components/Icons'
import { EFFORTS, effortLimits, isLevel, type EffortKind } from '../components/Effort'
import { Glyph } from '../components/Glyph'
import { Inline, Markdown, StreamingMarkdown } from '../components/Markdown'
import { proofLook, researchLook, Square, type Variant } from '../components/Square'
import { useDropTarget } from '../drop'
import { ago, count, duration } from '../format'
import { MODES } from '../modes'
import { go, href } from '../router'
import { useApp, useTick } from '../store'
import { BusyNote, Thinking, toolLine } from './shared'

type ChatMode = 'critic' | 'explore' | 'free'

const INTRO: Record<ChatMode, { title: string; text: string; examples: string[] }> = {
  free: {
    title: 'What are you working on?',
    text: 'Ask a question, share an argument, or describe what you want to work on. The assistant can answer directly, find precise references for known results, call on focused help when needed, and use the results to continue the discussion.',
    examples: [
      'Prove that every bounded sequence in $H^1(0,1)$ has a subsequence converging strongly in $L^2(0,1)$.',
      'Find a reference on the $H^2$ regularity of elliptic problems with Neumann boundary conditions.',
      'Review manuscript.tex, and find the literature on compact Sobolev embeddings it should cite.',
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

// How a card names what the model started. Critic and explore are both an answer here;
// a literature check also answers here, with its evidence.
type Kind = 'answer' | 'check' | 'prove' | 'literature' | 'referee' | 'writeup'
// `cover` names the mark (components/Glyph.tsx) and its colour class.
const KINDS: Record<Kind, { noun: string; label: string; cover: 'free' | 'check' | 'prove' | 'literature' | 'referee' | 'writeup' }> = {
  answer: { noun: 'an answer', label: 'Answer', cover: 'free' },
  check: { noun: 'a literature check', label: 'Literature check', cover: 'check' },
  prove: { noun: 'a proof', label: 'Proof', cover: 'prove' },
  literature: { noun: 'a literature report', label: 'Literature report', cover: 'literature' },
  referee: { noun: 'a review', label: 'Review', cover: 'referee' },
  writeup: { noun: 'a write-up', label: 'Write-up', cover: 'writeup' },
}
const kindOf = (mode: RouteMode): Kind => (mode === 'critic' || mode === 'explore' ? 'answer' : mode)
const effortKind = (mode: RouteMode): EffortKind | null =>
  mode === 'prove' ? 'proof' : mode === 'writeup' ? 'writeup' : mode === 'literature' || mode === 'referee' ? 'research' : null
const effortLabel = (effort?: string) => EFFORTS.find((item) => item.id === (isLevel(effort) ? effort : 'medium'))!.label

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
  const [resuming, setResuming] = useState(false)
  const [think, setThink] = useState<boolean>(app.status?.defaults.think ?? true)
  // Online search is on for new conversations unless the interface was launched --offline.
  const [online, setOnline] = useState<boolean>(app.status ? app.status.online : true)
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
  const paused = free && assistantResumable(data)
  const send = async (text?: string) => {
    const content = (text ?? draft).trim()
    if (!content || sending || resuming || paused || (free && (routing || ours)) || (!free && busy)) return
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
  const resume = async () => {
    if (!id || resuming) return
    setResuming(true)
    setSendError('')
    try { await api.resumeAssistant(id) } catch (reason) { setSendError((reason as Error).message) } finally { setResuming(false) }
  }
  const transcript = data?.transcript || []
  const last = [...transcript].reverse().find((item) => item.role !== 'notice' && item.role !== 'route')
  const canReview = !!id && !busy && !paused && !!last && last.role === 'assistant' && !!last.content
  // Another conversation's work can be queued; this conversation has one active turn.
  const blocked = free ? routing || ours || paused : busy

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
          {routing && !live && (
            <div className="entry message message-model">
              <span className="in-margin speaker speaker-model mode-free">∀</span>
              <p className="muted routing"><Square variant="running" size={12} /> Reading your request…</p>
            </div>
          )}
          {live && <LiveTurnView live={live} mode={mode} />}
          {free && assistantResumable(data) && !ours && (
            <div className="entry message message-model">
              <span className="in-margin speaker speaker-model mode-free">{MODES.free.glyph}</span>
              <p className="notice">The assistant is paused. Resume with the remaining budget, or start a new conversation for another request.</p>
            </div>
          )}
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
              <PauseIcon size={16} /> {free ? 'Pause' : 'Stop'}
            </button>
          ) : (
            <button type="submit" className="btn btn-primary composer-send" disabled={blocked || sending || resuming || !draft.trim()} aria-label="Send">
              <SendIcon size={16} /> Send
            </button>
          )}
        </div>
        <div className="composer-bar">
          {free ? <span className="muted small" title="Short questions get a direct answer; harder tasks receive more effort as needed.">Adaptive effort</span>
            : <Toggle label="Thinking" checked={think} onChange={changeThink} />}
          <span title={locked ? 'Launched with --offline: online search stays off' : online ? 'Search queries leave this computer' : 'Local and cached sources only'}>
            <Toggle label="Search online" checked={online && !locked} onChange={changeOnline} disabled={locked} />
          </span>
          {free && assistantResumable(data) && (
            <button type="button" className="btn btn-quiet btn-small" disabled={busy || resuming || sending} onClick={resume}>
              <PlayIcon size={14} /> {resuming ? 'Resuming…' : 'Resume assistant'}
            </button>
          )}
          {free && data?.assistant_budget && (
            <span className="muted small" title="Shared allowance for the assistant and all focused tasks in this turn">
              {count(data.assistant_budget.tokens.used)} / {count(data.assistant_budget.tokens.limit)} tokens
              {' · '}{duration(data.assistant_budget.seconds.used)} / {duration(data.assistant_budget.seconds.limit)}
            </span>
          )}
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
  // An Assistant conversation answers with one voice, whichever engine mode wrote it, except
  // a literature check, which keeps its citation mark; older critique and exploration chats
  // keep their own mark.
  let mark: ChatMode | 'check' = mode
  const flush = (key: number) => {
    if (!blocks.length) return
    out.push(
      <div key={'answer' + key} className="entry message message-model">
        <span className={`in-margin speaker speaker-model mode-${mark}`} title={mark === 'check' ? 'Literature check' : MODES[mark].label}><Glyph mode={mark} /></span>
        <div className="message-body">{blocks}</div>
      </div>,
    )
    blocks = []
  }
  const markFor = (item: TranscriptItem): ChatMode | 'check' => (item.mode === 'check' ? 'check' : mode)
  for (let i = 0; i < steps.length; i++) {
    const { item, index } = steps[i]
    if (item.role === 'route') {
      flush(index)
      out.push(<RouteCard key={'route' + index} chatId={chatId} index={index} item={item} />)
    } else if (item.role === 'assistant') {
      if (markFor(item) !== mark) { flush(index); mark = markFor(item) }
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

/** What the model started for a message, at which effort, with a way to cancel it. */
function RouteCard({ chatId, index, item }: { chatId: string; index: number; item: TranscriptItem }) {
  const app = useApp()
  const [error, setError] = useState('')
  const [acting, setActing] = useState(false)
  useEffect(() => { setError(item.error || '') }, [item.error])
  if (item.status === 'dismissed') {
    return <div className="entry route-dismissed"><span className="in-margin" /><span className="muted small">Suggestion dismissed.</span></div>
  }
  if (item.mode === 'clarify' || !item.mode) {
    return (
      <div className="entry route route-clarify">
        <span className="in-margin speaker speaker-model mode-free">{MODES.free.glyph}</span>
        <div className="route-card">
          <p className="route-title">One detail first</p>
          <Markdown className="route-question">{item.question || ''}</Markdown>
          <p className="muted small">Answer in the message box; the work starts as soon as the request is clear.</p>
        </div>
      </div>
    )
  }
  const mode = item.mode as RouteMode
  const kind = kindOf(mode)
  const info = item.orchestrated && mode === 'critic'
    ? { noun: 'a critique', label: 'Critique', cover: 'critic' as const }
    : item.orchestrated && mode === 'explore'
      ? { noun: 'an exploration', label: 'Exploration', cover: 'explore' as const }
      : KINDS[kind]
  const effort = kind === 'answer' ? '' : `${effortLabel(item.effort)} effort`
  const act = async (action: () => Promise<unknown>) => {
    setActing(true)
    setError('')
    try { await action() } catch (reason) { setError((reason as Error).message) } finally { setActing(false) }
  }
  // Suggestions saved before jobs started on their own, and jobs that failed or were
  // cancelled before running, can be started (again) as the model chose them.
  const start = () => act(() => {
    const limits = effortKind(mode) && app.status ? effortLimits(effortKind(mode)!, app.status.defaults, isLevel(item.effort) ? item.effort : 'medium') : undefined
    return api.startRoute(chatId, index, { mode, request: item.request || '', files: item.files || [], limits: limits as Record<string, number> | undefined })
  })

  const task = app.snapshot.task
  const { queued, running } = routeActivity(item, app.snapshot)
  let status = ''
  let variant: Variant = 'ready'
  if (item.job_id && mode === 'prove') {
    const job = app.proofs?.find((proof) => proof.id === item.job_id)
    if (job) ({ variant, label: status } = proofLook(job.status, job.running && (!item.orchestrated || running)))
  } else if (item.job_id) {
    const job = app.research?.find((report) => report.id === item.job_id)
    if (job) ({ variant, label: status } = researchLook(job.status, job.running && (!item.orchestrated || running)))
  }
  let headline: string
  if (item.status === 'proposed') { headline = `${info.label} suggested`; status = 'Not started'; variant = 'ready' }
  else if (item.status === 'failed') { headline = item.orchestrated ? `${info.label} did not finish` : `Could not start ${info.noun}`; variant = 'error' }
  else if (item.status === 'cancelled') { headline = `${info.label} cancelled`; status = 'Removed from the queue'; variant = 'spent' }
  else if (queued) { headline = `${info.label} queued`; status = 'Starts when the model is free'; variant = 'ready' }
  else if (running && task?.state === 'pausing') { headline = `Stopping ${info.noun}`; status = 'At the next checkpoint'; variant = 'running' }
  else if (running) { headline = kind === 'answer' && !item.orchestrated ? 'Answering here' : kind === 'check' && !item.orchestrated ? 'Checking references' : `${item.job_id || item.orchestrated ? 'Running' : 'Starting'} ${info.noun}`; status = ''; variant = 'running' }
  else if (item.status === 'stopped') { headline = kind === 'answer' && !item.orchestrated ? 'The answer was stopped' : `${info.label} stopped`; status = status || 'Paused'; variant = 'paused' }
  else if (item.orchestrated && item.status === 'done') { headline = `${info.label} returned`; if (!item.job_id) { status = ''; variant = 'ready' } }
  else if (item.orchestrated) { headline = `${info.label} awaiting resume`; status = 'Paused'; variant = 'paused' }
  else if (kind === 'answer') { headline = 'Answered below'; status = ''; variant = 'complete' }
  else if (kind === 'check') { headline = 'References checked below'; status = ''; variant = 'complete' }
  else { headline = info.label }
  return (
    <div className={'entry route route-' + (item.status || 'proposed')}>
      <span className={`in-margin speaker speaker-model mode-${info.cover}`} title={info.label}><Glyph mode={info.cover} /></span>
      <div className="route-card route-auto">
        <div className="route-line">
          <Square variant={variant} size={14} />
          <p className="route-title">{headline}</p>
          {effort && <span className="route-effort">{effort}</span>}
          {status && <span className="route-status">{status}</span>}
        </div>
        <div className="route-actions">
          {(queued || running) && (
            <button type="button" className="btn btn-small" disabled={acting || task?.state === 'pausing'}
              title={item.orchestrated ? 'Pause the assistant and its current task; resume with the remaining budget' : queued ? 'Take it out of the queue' : 'Stop at the next checkpoint; the job is kept and can be resumed from its page'}
              onClick={() => act(() => item.orchestrated ? api.pause() : api.cancelRoute(chatId, index))}>
              {item.orchestrated ? <PauseIcon size={14} /> : <CloseIcon size={14} />} {item.orchestrated ? 'Pause assistant' : 'Cancel'}
            </button>
          )}
          {item.job_id && mode !== 'check' && <a className="btn btn-small btn-quiet" href={href(mode, item.job_id)}>Open the {info.label.toLowerCase()}</a>}
          {!item.orchestrated && ['proposed', 'failed', 'cancelled'].includes(item.status || '') && (
            <button type="button" className="btn btn-small" disabled={acting} onClick={start}>
              {item.status === 'proposed' ? 'Start' : 'Start again'}
            </button>
          )}
          <Collapse className="route-details" summary={(kind === 'answer' || kind === 'check') && !item.orchestrated ? 'Question as the model sees it' : `Request given to the ${info.label.toLowerCase()}`}>
            <div className="route-summary"><Inline limit={1200}>{item.request || ''}</Inline></div>
            {item.files && item.files.length > 0 && (
              <div className="attachments">{item.files.map((path) => <span key={path} className="chip chip-static"><FileIcon size={13} /><span>{path}</span></span>)}</div>
            )}
            {item.missing && item.missing.length > 0 && <p className="muted small">Not found in this folder: {item.missing.join(', ')}.</p>}
            {item.reason && <p className="muted small">Why: {item.reason}</p>}
          </Collapse>
        </div>
        {item.outcome && item.outcome !== error && <p className="muted small">{item.outcome}</p>}
        <ErrorNote>{error}</ErrorNote>
      </div>
    </div>
  )
}

function LiveTurnView({ live, mode }: { live: LiveTurn; mode: ChatMode }) {
  const glyph: ChatMode | 'check' = live.mode === 'check' ? 'check' : mode
  return (
    <div className="turn turn-live">
      {live.kind === 'message' && live.user && <UserMessage item={{ role: 'user', content: live.user, time: '' }} />}
      <div className="entry message message-model">
        <span className={`in-margin speaker speaker-model mode-${glyph}`}><Glyph mode={glyph} /></span>
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
