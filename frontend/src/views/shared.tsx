import { useEffect, useRef, useState, type ReactNode } from 'react'
import { api, type Artifact, type Review, type ToolCall } from '../api'
import { Collapse, ErrorNote, Loading, Modal } from '../components/common'
import { BackIcon, PanelRightIcon } from '../components/Icons'
import { Inline, Markdown, StreamingMarkdown } from '../components/Markdown'
import { taskHref } from '../components/Shell'
import { issueLook, Square } from '../components/Square'
import { bytes, count, firstLine } from '../format'
import { setPanelHidden, usePanelHidden } from '../panels'
import { onStream, useApp } from '../store'

/** Says what the model is doing; `queue` explains that new work waits its turn. */
export function BusyNote({ queue = false }: { queue?: boolean }) {
  const app = useApp()
  const task = app.snapshot.task
  if (!task) return null
  const waiting = app.snapshot.queue.length
  return (
    <p className="busy-note">
      The model is working on a {task.label.toLowerCase()}: <a href={taskHref(task, app)}><Inline limit={70}>{withoutStop(task.title)}</Inline></a>.
      {queue
        ? ` This one will wait in the queue${waiting ? `, after ${waiting} other ${waiting === 1 ? 'job' : 'jobs'}` : ''}, and start when the model is free.`
        : ' Pause it first: the model serves one task at a time, and time budgets count wall-clock time.'}
    </p>
  )
}

/** A title quoted inside a sentence, without its own final full stop. */
const withoutStop = (title: string) => title.trim().replace(/[.。]+$/, '')

export function queuedMessage(title: string) {
  return `Added to the queue: “${firstLine(withoutStop(title), 80)}”. It starts when the model is free. The queue is listed under the running job, where you can cancel it.`
}

/** The job's overview column (budget, candidates, checks); it slides away to a spine. */
export function JobAside({ label, children }: { label: string; children: ReactNode }) {
  const hidden = usePanelHidden('aside')
  if (hidden) {
    return (
      <aside className="job-aside" aria-label={label}>
        <button type="button" className="spine aside-spine" onClick={() => setPanelHidden('aside', false)}
          title={`Show the ${label.toLowerCase()}`} aria-label={`Show the ${label.toLowerCase()}`}>
          <BackIcon size={16} />
          <span className="spine-text">{label}</span>
        </button>
      </aside>
    )
  }
  return (
    <aside className="job-aside" aria-label={label}>
      <div className="aside-inner">
        <div className="aside-bar">
          <span className="aside-name">{label}</span>
          <button type="button" className="icon-btn" title="Hide this panel" aria-label={`Hide the ${label.toLowerCase()}`}
            onClick={() => setPanelHidden('aside', true)}>
            <PanelRightIcon />
          </button>
        </div>
        {children}
      </div>
    </aside>
  )
}

/** Class for a job page, so its grid gives the overview column its full width or a spine. */
export function useJobClass() {
  return 'job' + (usePanelHidden('aside') ? ' job-aside-hidden' : '')
}

export const ROLES: Record<string, { active: string; name: string; structured?: boolean }> = {
  solver: { active: 'The solver is writing a complete proof', name: 'Solver' },
  verifier: { active: 'A fresh verifier is reviewing the whole proof', name: 'Verifier', structured: true },
  // Roles of legacy v0.4 proof jobs, whose saved files stay readable
  checkpoint: { active: 'Turning a truncated attempt into a checkpoint', name: 'Checkpoint' },
  critic: { active: 'A fresh critic is checking the attempt', name: 'Critic', structured: true },
  recorder: { active: 'The recorder is updating the ledger', name: 'Recorder', structured: true },
  auditor: { active: 'The auditor is checking the whole proof', name: 'Auditor', structured: true },
  planner: { active: 'The planner is choosing a different approach', name: 'Planner', structured: true },
  plan: { active: 'Planning the investigation', name: 'Plan' },
  investigate: { active: 'Investigating sources', name: 'Investigation' },
  draft: { active: 'Drafting the report', name: 'Draft' },
  review: { active: 'A fresh reviewer is checking the draft', name: 'Review' },
  fidelity: { active: 'A fresh reviewer is checking the write-up against your notes', name: 'Review' },
  revise: { active: 'Revising the report', name: 'Revision' },
  outline: { active: 'Planning the sections from your notes', name: 'Outline', structured: true },
  section: { active: 'Writing a section from your notes', name: 'Section' },
  repair: { active: 'Repairing the sections named in LaTeX errors', name: 'Repair' },
  // Review variants (quick check, journal report, explanation): small structured calls.
  check: { active: 'A fresh verifier is checking one proof', name: 'Check', structured: true },
  confirm: { active: 'An independent re-check of an alleged issue', name: 'Re-check', structured: true },
  extract: { active: 'Finding the statement and the proof', name: 'Extract', structured: true },
  pick: { active: 'Finding the result the request names', name: 'Pick', structured: true },
  read: { active: 'Reading a part of the paper: what it does, typos and presentation', name: 'Reading', structured: true },
  scope: { active: 'Writing an overview of the paper from the reading notes', name: 'Overview', structured: true },
  novelty: { active: 'Comparing the contribution with the search results', name: 'Novelty', structured: true },
  presentation: { active: 'Reading for presentation', name: 'Presentation', structured: true },
  write: { active: 'Writing the prose of the report', name: 'Write', structured: true },
  explain: { active: 'Writing the explanation', name: 'Explanation', structured: true },
  fill: { active: 'Explaining the lines still missing', name: 'Fill', structured: true },
  'explain-check': { active: 'Checking the explanation against the source', name: 'Accuracy check', structured: true },
  'explain-fix': { active: 'Correcting the explanation', name: 'Correction', structured: true },
}

export type LiveText = { text: string; thinking: string; tool_calls: ToolCall[]; done: boolean; loaded: boolean }

/** Follow a model stream: load what is saved, then append the watcher's deltas. */
export function useLiveStream(job: 'proof' | 'research', id: string, file: string | null): LiveText {
  const [value, setValue] = useState<LiveText>({ text: '', thinking: '', tool_calls: [], done: false, loaded: false })
  const ref = useRef({ end: 0, loading: false, stale: false })
  useEffect(() => {
    setValue({ text: '', thinking: '', tool_calls: [], done: false, loaded: false })
    if (!file) return
    let alive = true
    let acc: LiveText = { text: '', thinking: '', tool_calls: [], done: false, loaded: false }
    ref.current = { end: 0, loading: false, stale: false }
    const fetchFrom = async (from: number) => {
      if (ref.current.loading) { ref.current.stale = true; return }
      ref.current.loading = true
      try {
        const chunk = job === 'proof' ? await api.proofStream(id, file, from) : await api.researchStream(id, file, from)
        if (!alive) return
        if (chunk.reset || from === 0) acc = { text: '', thinking: '', tool_calls: [], done: false, loaded: true }
        acc = { text: acc.text + chunk.text, thinking: acc.thinking + chunk.thinking,
          tool_calls: [...acc.tool_calls, ...chunk.tool_calls], done: acc.done || chunk.done, loaded: true }
        ref.current.end = chunk.end
        setValue(acc)
      } catch {
        /* the next event or state change retries */
      } finally {
        ref.current.loading = false
        if (alive && ref.current.stale) { ref.current.stale = false; fetchFrom(ref.current.end) }
      }
    }
    fetchFrom(0)
    const off = onStream(job, id, file, (chunk) => {
      if (chunk.end <= ref.current.end) return
      if (chunk.start === ref.current.end && !ref.current.loading) {
        acc = { text: acc.text + chunk.text, thinking: acc.thinking + chunk.thinking,
          tool_calls: [...acc.tool_calls, ...chunk.tool_calls], done: acc.done || chunk.done, loaded: true }
        ref.current.end = chunk.end
        setValue(acc)
      } else {
        fetchFrom(ref.current.end)
      }
    })
    return () => { alive = false; off() }
  }, [job, id, file])
  return value
}

export function toolLine(call: ToolCall): string {
  const fn = call.function || { name: '?', arguments: {} }
  let args: Record<string, unknown> = {}
  if (typeof fn.arguments === 'string') {
    try { args = JSON.parse(fn.arguments) } catch { return `${fn.name} ${fn.arguments}` }
  } else args = fn.arguments || {}
  const parts = Object.entries(args).map(([key, value]) => `${key} ${typeof value === 'string' ? value : JSON.stringify(value)}`)
  return `${fn.name}${parts.length ? ': ' + parts.join(', ') : ''}`
}

export function Thinking({ text, live }: { text: string; live?: boolean }) {
  if (!text) return null
  return (
    <Collapse className="thinking" summary={<>{live ? 'Thinking' : 'Thought'} <span className="muted">{count(text.length)} characters</span></>}>
      <Markdown className="thinking-text">{text}</Markdown>
    </Collapse>
  )
}

function prettyJson(text: string): string | null {
  try {
    return JSON.stringify(JSON.parse(text), null, 2)
  } catch {
    return null
  }
}

const VERDICTS: Record<string, string> = {
  no_issue_found: 'No issue found', issues_found: 'Issues found', uncertain: 'Uncertain',
}

/** A verifier's answer: its verdict, explanation and located issues. */
export function ReviewBody({ response }: { response: NonNullable<Review['response']> }) {
  return (
    <div className="review-body">
      <p className="audit-verdict">Verdict: {VERDICTS[response.verdict] || response.verdict}</p>
      {response.explanation && <Markdown>{response.explanation}</Markdown>}
      {response.issues.map((issue, index) => {
        const look = issueLook(issue.kind)
        return (
          <div key={index} className={issue.kind === 'uncertainty' ? 'missing' : 'objection'}>
            <p className={issue.kind === 'uncertainty' ? 'missing-label' : 'objection-label'}>
              <Square variant={look.variant} size={11} /> {look.label}{issue.location && <>: <Inline limit={200}>{issue.location}</Inline></>}
            </p>
            <Markdown>{issue.evidence}</Markdown>
          </div>
        )
      })}
    </div>
  )
}

function reviewResponse(value: unknown): NonNullable<Review['response']> | null {
  const item = value as Review['response']
  return item && typeof item === 'object' && typeof item.verdict === 'string' && typeof item.explanation === 'string'
    && Array.isArray(item.issues) ? item : null
}

/** Model output as text: prose is typeset; structured roles show their JSON. */
export function StreamBody({ stream, role, live }: { stream: LiveText | NonNullable<Artifact['stream']>; role: string; live?: boolean }) {
  const structured = ROLES[role]?.structured
  const pretty = structured ? prettyJson(stream.text) : null
  const review = role === 'verifier' && pretty ? reviewResponse(JSON.parse(stream.text)) : null
  return (
    <div className="stream-body">
      <Thinking text={stream.thinking} live={live && !stream.text} />
      {structured ? (
        review ? <ReviewBody response={review} /> :
        pretty ? <StructuredView value={JSON.parse(stream.text)} /> :
          stream.text ? <pre className="json-live">{stream.text}{live ? '▍' : ''}</pre> : null
      ) : live ? (
        <StreamingMarkdown text={stream.text} />
      ) : (
        <Markdown>{stream.text}</Markdown>
      )}
      {stream.tool_calls.length > 0 && (
        <ul className="tool-list">
          {stream.tool_calls.map((call, index) => <li key={index} className="tool-line">{toolLine(call)}</li>)}
        </ul>
      )}
      {live && !stream.text && !stream.thinking && <p className="muted small">Waiting for the first tokens. The model may still be loading into memory.</p>}
    </div>
  )
}

/** Critic, recorder, planner and auditor answers, laid out by field. */
export function StructuredView({ value }: { value: unknown }) {
  if (!value || typeof value !== 'object') return <pre className="code-block">{JSON.stringify(value, null, 2)}</pre>
  const entries = Object.entries(value as Record<string, unknown>).filter(([key]) => !key.startsWith('_'))
  return (
    <dl className="structured">
      {entries.map(([key, item]) => (
        <div key={key} className="structured-row">
          <dt>{key.replace(/_/g, ' ')}</dt>
          <dd>
            {typeof item === 'string' ? (item ? <Markdown>{item}</Markdown> : <span className="muted">empty</span>)
              : typeof item === 'boolean' ? (item ? 'yes' : 'no')
              : Array.isArray(item) && item.every((x) => typeof x === 'string') ? (
                item.length ? <ul>{item.map((x, i) => <li key={i}><Markdown>{x as string}</Markdown></li>)}</ul> : <span className="muted">none</span>
              ) : <pre className="code-block">{JSON.stringify(item, null, 2)}</pre>}
          </dd>
        </div>
      ))}
    </dl>
  )
}

type Message = { role: string; content?: string; tool_calls?: ToolCall[]; tool_name?: string }

function RequestView({ payload }: { payload: Record<string, unknown> }) {
  const messages = (payload.messages as Message[]) || []
  const options = (payload.options as Record<string, unknown>) || {}
  const tools = (payload.tools as { function: { name: string } }[]) || []
  return (
    <div className="request">
      <p className="muted">
        Model {String(payload.model)}; context {String(options.num_ctx ?? '?')}; output limit {String(options.num_predict ?? '?')};
        thinking {payload.think ? 'on' : 'off'}{payload.format ? '; structured answer required' : ''}
        {tools.length ? `; tools: ${tools.map((tool) => tool.function.name).join(', ')}` : ''}.
      </p>
      {messages.map((message, index) => (
        <Collapse key={index} open={index === messages.length - 1}
          summary={<><strong>{message.role}</strong> <span className="muted">{count((message.content || '').length)} characters</span></>}>
          {message.tool_calls?.length ? <ul className="tool-list">{message.tool_calls.map((c, i) => <li key={i} className="tool-line">{toolLine(c)}</li>)}</ul> : null}
          <pre className="plain">{message.content}</pre>
        </Collapse>
      ))}
    </div>
  )
}

export function ArtifactViewer({ job, id, name, onClose }: { job: 'proof' | 'research'; id: string; name: string; onClose: () => void }) {
  const [artifact, setArtifact] = useState<Artifact | null>(null)
  const [error, setError] = useState('')
  const [raw, setRaw] = useState(false)
  useEffect(() => {
    const load = job === 'proof' ? api.proofArtifact(id, name) : api.researchArtifact(id, name)
    load.then(setArtifact, (reason: Error) => setError(reason.message))
  }, [job, id, name])
  const role = name.replace(/^\d+-/, '').replace(/(-request|-stream)?\.(md|jsonl)$/, '')
  const json = artifact?.json as Record<string, unknown> | undefined
  return (
    <Modal title={name} onClose={onClose} wide>
      <ErrorNote>{error}</ErrorNote>
      {!artifact && !error && <Loading />}
      {artifact && (
        <>
          <div className="artifact-bar">
            <span className="muted">{bytes(artifact.size)}; saved in the job's artifacts folder</span>
            <button type="button" className="btn btn-small btn-quiet" onClick={() => setRaw(!raw)}>{raw ? 'Formatted view' : 'Raw file'}</button>
          </div>
          {raw ? <pre className="plain">{artifact.text}</pre>
            : artifact.stream ? <StreamBody stream={artifact.stream} role={role} />
            : role === 'proof-candidate' ? <Markdown>{artifact.text}</Markdown>
            : json && Array.isArray(json.messages) ? <RequestView payload={json} />
            : json && typeof json.text === 'string' ? <StreamBody stream={{ text: json.text as string, thinking: (json.thinking as string) || '', tool_calls: (json.calls as ToolCall[]) || [], done: true, stats: {} }} role="solver" />
            : json && 'result' in json ? (
              <>
                <p><strong>{String(json.name ?? '')}</strong> <code>{JSON.stringify(json.arguments)}</code></p>
                <pre className="plain">{String(json.result)}</pre>
              </>
            )
            : json ? <pre className="plain">{JSON.stringify(json, null, 2)}</pre>
            : <pre className="plain">{prettyJson(artifact.text) || artifact.text}</pre>}
        </>
      )}
    </Modal>
  )
}

// Lines the phase row and header already show; the log keeps the rest.
const REDUNDANT = /^(Proof [0-9a-f-]{36}:|Research [0-9a-f-]{36}:|Research phase:)/

export function ActivityLog({ items }: { items: { kind: string; text: string }[] }) {
  const shown = items.filter((item) => !(item.kind === 'notice' && REDUNDANT.test(item.text))).slice(-5)
  if (!shown.length) return null
  return (
    <ul className="activity">
      {shown.map((item, index) => (
        <li key={index} className={item.kind === 'tool' ? 'tool-line' : item.kind === 'result' ? 'tool-result-live' : ''}>
          {firstLine(item.text, 200)}
        </li>
      ))}
    </ul>
  )
}
