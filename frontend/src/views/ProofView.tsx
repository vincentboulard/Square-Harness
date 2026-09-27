import { useEffect, useState } from 'react'
import { api, type Call, type Candidate, type Pending, type ProofDetail, type Review } from '../api'
import { Collapse, ErrorNote, Loading, Meter, SourceLines, Tabs, useLoad } from '../components/common'
import { PauseIcon, PlayIcon } from '../components/Icons'
import { Inline, Markdown } from '../components/Markdown'
import { candidateLook, proofLook, proofResumable, Square, type Variant } from '../components/Square'
import { ago, bytes, count, duration, plural } from '../format'
import { go } from '../router'
import { useApp, useTick } from '../store'
import { ActivityLog, ArtifactViewer, BusyNote, JobAside, ReviewBody, ROLES, StreamBody, useJobClass, useLiveStream } from './shared'

const KINDS: Record<string, string> = {
  initial: 'initial solve', repair: 'repair', continue: 'continuation', retry: 'fresh solve',
}
const PHASES: Record<string, string> = { solve: 'Solve', review: 'Review', repair: 'Repair', continue: 'Continuation' }
const SELECTIONS: Record<string, string> = {
  first_written_candidate: 'first written answer',
  finished_response_replaces_fragment: 'a finished answer replaces an unfinished one',
  whole_proof_review_found_no_issue: 'the whole-proof review found no issue',
}

type Attempt = { index: number; candidate?: Candidate; reviews: Review[]; calls: Call[]; pending: Pending | null }

function attempts(data: ProofDetail): Attempt[] {
  const last = Math.max(data.budget.rounds.used, ...data.candidates.map((c) => c.attempt), data.pending?.index || 0)
  return Array.from({ length: last }, (_, i) => {
    const candidate = data.candidates.find((c) => c.attempt === i + 1)
    return {
      index: i + 1, candidate,
      reviews: candidate ? data.reviews.filter((review) => review.candidate === candidate.id) : [],
      calls: data.calls.filter((call) => call.round === i + 1),
      pending: data.pending && data.pending.index === i + 1 ? data.pending : null,
    }
  })
}

export function ProofView({ id, tab }: { id: string; tab: string | null }) {
  const tick = useTick(id)
  const app = useApp()
  const { data, error } = useLoad(() => api.proof(id), [id, tick])
  const [artifact, setArtifact] = useState<string | null>(null)
  const active = tab || 'work'
  const jobClass = useJobClass()
  const task = app.snapshot.task
  const ours = !!task && task.kind === 'proof' && task.target === id && ['starting', 'running', 'pausing'].includes(task.state)

  const focusAttempt = (index: number) => {
    if (active !== 'work') go('prove', id, null)
    window.setTimeout(() => document.getElementById('attempt-' + index)?.scrollIntoView({ block: 'start', behavior: 'smooth' }), 60)
  }

  if (!data) {
    return <div className="sheet paper"><div className="sheet-inner">{error ? <ErrorNote>{error}</ErrorNote> : <Loading />}</div></div>
  }
  return (
    <div className={jobClass}>
      <div className="job-main sheet paper">
        <div className="sheet-inner">
          <ProofHeader data={data} ours={ours} pausing={task?.state === 'pausing'} />
          <Tabs active={active} onChange={(next) => go('prove', id, next === 'work' ? null : next)} tabs={[
            { id: 'work', label: 'Work', badge: data.legacy ? undefined : data.budget.rounds.used || undefined },
            { id: 'overview', label: 'Overview', narrowOnly: true },
            { id: 'report', label: 'Report' },
            { id: 'sources', label: 'Sources', badge: data.sources.length || undefined },
            { id: 'files', label: 'Files', badge: data.artifacts.length || undefined },
          ]} />
          {active === 'work' && (data.legacy ? <LegacyWork data={data} /> : <Work data={data} ours={ours} onArtifact={setArtifact} />)}
          {active === 'overview' && <div className="ledger-tab"><Overview data={data} onAttempt={focusAttempt} /></div>}
          {active === 'report' && <ProofReport id={id} tick={tick} />}
          {active === 'sources' && <ProofSources id={id} goal={data.goal} />}
          {active === 'files' && <Files names={data.artifacts} onOpen={setArtifact} />}
        </div>
      </div>
      <JobAside label="Overview">
        <Overview data={data} onAttempt={focusAttempt} />
      </JobAside>
      {artifact && <ArtifactViewer job="proof" id={id} name={artifact} onClose={() => setArtifact(null)} />}
    </div>
  )
}

function ProofHeader({ data, ours, pausing }: { data: ProofDetail; ours: boolean; pausing: boolean }) {
  const look = proofLook(data.status, data.running, data.version)
  const [error, setError] = useState('')
  const [expanded, setExpanded] = useState(false)
  const app = useApp()
  const busy = !!app.snapshot.task && ['starting', 'running', 'pausing'].includes(app.snapshot.task.state)
  const queued = app.snapshot.queue.some((item) => item.target === data.id)
  const long = data.goal.length > 420
  const resumable = proofResumable(data.status, data.version)
  const settings = data.settings
  const act = (action: Promise<unknown>) => { setError(''); action.catch((reason: Error) => setError(reason.message)) }
  return (
    <header className="job-head">
      <div className="job-status">
        <Square variant={look.variant} size={18} label={look.label} />
        <span className="job-status-label">{look.label}</span>
        {data.stop_reason && !data.running && <span className="job-stop">{data.stop_reason}</span>}
      </div>
      <div className={'job-goal' + (long && !expanded ? ' job-goal-clamped' : '')}>
        <Markdown>{data.goal}</Markdown>
      </div>
      {long && <button type="button" className="link-btn" onClick={() => setExpanded(!expanded)}>{expanded ? 'Show less' : 'Show the whole request'}</button>}
      <div className="job-facts">
        {data.sources.map((source) => <span key={source.path} className="fact-file">{source.path}</span>)}
        <span>{String(settings.model)}</span>
        <span>Context {count(settings.ctx)}</span>
        {!data.legacy && settings.max_predict !== undefined && <span>Solve up to {count(settings.max_predict)} tokens</span>}
        {!data.legacy && settings.verify_tokens !== undefined && <span>Review up to {count(settings.verify_tokens)} tokens</span>}
        {data.legacy && <span>v0.4 engine</span>}
        <span>Started {ago(data.created_at)}</span>
      </div>
      {data.legacy && (
        <p className="warning-note">
          This proof was made by the earlier v0.4 engine. Its report, sources and files stay readable here, but v0.5 cannot
          resume it: start a new proof to continue the work.
        </p>
      )}
      {data.recovery_notice && <p className="warning-note">{data.recovery_notice}</p>}
      <div className="job-actions">
        {ours ? (
          <button type="button" className="btn" disabled={pausing} onClick={() => act(api.pause())}>
            <PauseIcon size={16} /> {pausing ? 'Pausing…' : 'Pause'}
          </button>
        ) : resumable ? (
          <button type="button" className="btn btn-primary" disabled={queued} onClick={() => act(api.resumeProof(data.id, busy))}>
            <PlayIcon size={16} /> {queued ? 'Waiting in the queue' : busy ? 'Resume when the model is free' : 'Resume with the remaining budget'}
          </button>
        ) : null}
        {data.running && !ours && <span className="muted">Running in another process, such as a terminal.</span>}
      </div>
      {!ours && busy && resumable && !queued && <BusyNote queue />}
      <ErrorNote>{error}</ErrorNote>
    </header>
  )
}

function Work({ data, ours, onArtifact }: { data: ProofDetail; ours: boolean; onArtifact: (name: string) => void }) {
  useEffect(() => {
    if (data.running) document.getElementById('live')?.scrollIntoView({ block: 'center' })
    // Only when the page opens on a running job.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const list = attempts(data)
  const selected = data.candidates.find((candidate) => candidate.id === data.selected_candidate)
  return (
    <div className="work">
      {selected && data.answer !== null && <SelectedAnswer data={data} candidate={selected} />}
      {selected && data.answer === null && (
        <p className="warning-note">The saved file of the selected answer {selected.id} no longer matches its recorded digest, so it is not shown.</p>
      )}
      {!list.length && <p className="empty-note">No attempt has started yet. Resume the job to begin.</p>}
      {list.map((attempt) => <AttemptEntry key={attempt.index} attempt={attempt} data={data} ours={ours} onArtifact={onArtifact} />)}
      {data.selection_history.length > 1 && (
        <p className="muted small">
          Selection: {data.selection_history.map((step) => `${step.selected} (${SELECTIONS[step.reason] || step.reason})`).join(' → ')}.
          The initial answer {data.initial_candidate} is always kept.
        </p>
      )}
    </div>
  )
}

function SelectedAnswer({ data, candidate }: { data: ProofDetail; candidate: Candidate }) {
  const clean = candidate.review_status === 'no_issue_found'
  const look = candidateLook(candidate)
  const review = [...data.reviews].reverse().find((item) => item.candidate === candidate.id && item.response)
  const title = clean ? 'Answer the review accepted'
    : candidate.review_status === 'issues_found' ? 'Selected answer, with open objections'
    : candidate.review_status === 'uncertain' ? 'Selected answer, review uncertain'
    : candidate.review_status === 'review_unavailable' ? 'Selected answer, no usable review'
    : !candidate.transport_complete ? 'Selected answer, unfinished' : 'Selected answer, not reviewed yet'
  return (
    <section className="entry candidate">
      <span className="in-margin"><Square variant={clean ? 'complete' : look.variant} size={18} label={look.label} /></span>
      <div>
        <h2 className="candidate-title">{title}</h2>
        <p className="caveat">
          {clean
            ? 'A fresh review by the same model found no issue in the whole proof. That is a model judgement, not verification: check every step yourself.'
            : candidate.review_status === 'unreviewed'
              ? 'This is the answer the harness exports as proof.md. It has not been reviewed yet.'
              : 'This is the answer the harness exports as proof.md. It has not passed a review; the attempts below show what the reviewer said.'}
          {' '}Candidate {candidate.id}, from the {KINDS[candidate.kind] || candidate.kind} in attempt {candidate.attempt}.
        </p>
        <Markdown className="candidate-text">{data.answer || ''}</Markdown>
        {clean && <p className="tombstone" aria-hidden="true">∎</p>}
        {clean && review?.response?.explanation && (
          <Collapse summary="Why the reviewer accepted it"><Markdown>{review.response.explanation}</Markdown></Collapse>
        )}
      </div>
    </section>
  )
}

function AttemptEntry({ attempt, data, ours, onArtifact }: { attempt: Attempt; data: ProofDetail; ours: boolean; onArtifact: (name: string) => void }) {
  const { index, candidate, reviews, calls, pending } = attempt
  const kind = candidate?.kind || pending?.kind || 'initial'
  const parent = kind === 'retry' ? null : candidate?.parent || pending?.parent
  // A finished job keeps its last pending step on record; only live or resumable work is current.
  const current = !!pending && (pending.phase === 'solve' || pending.phase === 'review')
    && (data.running || proofResumable(data.status, data.version))
  const live = current && data.running && data.live && data.live.round === index ? data.live : null
  const stopped = current && !data.running
  const phase = (name: 'solve' | 'review'): Variant =>
    (name === 'solve' ? !!candidate : reviews.length > 0 && !(current && pending?.phase === 'review')) ? 'complete'
      : current && pending?.phase === name ? (data.running ? 'running' : 'paused') : 'ready'
  const reviewed = !!candidate?.transport_complete || reviews.length > 0 || pending?.phase === 'review'
  return (
    <section id={'attempt-' + index} className={'entry round' + (current ? ' round-pending' : '')}>
      <span className="in-margin round-number">{index}</span>
      <div className="round-body">
        <div className="round-head">
          <h2>Attempt {index}<span className="round-kind">, {KINDS[kind] || kind}{parent ? ` of ${parent}` : ''}</span></h2>
          <ol className="phases" aria-label="Phases">
            <li className={'phase phase-' + phase('solve')}><Square variant={phase('solve')} size={10} />Solve</li>
            {reviewed && <li className={'phase phase-' + phase('review')}><Square variant={phase('review')} size={10} />Review</li>}
          </ol>
        </div>
        {kind === 'retry' && <p className="muted small">The saved work did not fit a useful continuation, so the original problem was solved afresh.</p>}
        {kind === 'continue' && pending?.notes_chars_total !== undefined && pending.index === index && (
          <p className="muted small">Continues from the written text and {count(pending.notes_chars_kept || 0)} of {count(pending.notes_chars_total)} characters of saved notes.</p>
        )}
        {candidate && <CandidateEntry id={data.id} candidate={candidate} calls={calls} selected={candidate.id === data.selected_candidate} onArtifact={onArtifact} />}
        {reviews.map((review) => <ReviewEntry key={review.id} review={review} />)}
        {live && <LiveCall id={data.id} role={live.role} file={live.file} ours={ours} />}
        {stopped && <p className="muted small">This attempt stopped in the {(PHASES[pending!.phase] || pending!.phase).toLowerCase()} phase. Resuming continues from its last saved checkpoint; saved calls are not repeated.</p>}
        {calls.length > 0 && (
          <Collapse summary={<>Model calls <span className="muted">{calls.length}</span></>}>
            <CallList calls={calls} onArtifact={onArtifact} />
          </Collapse>
        )}
      </div>
    </section>
  )
}

function CandidateEntry({ id, candidate, calls, selected, onArtifact }: {
  id: string; candidate: Candidate; calls: Call[]; selected: boolean; onArtifact: (name: string) => void
}) {
  const [open, setOpen] = useState(false)
  const call = calls.find((item) => item.key === candidate.call)
  const stream = useLiveStream('proof', id, open && call ? call.stream : null)
  const look = candidateLook(candidate)
  return (
    <details className="collapse attempt" onToggle={(event) => setOpen((event.currentTarget as HTMLDetailsElement).open)}>
      <summary>
        <Square variant={look.variant} size={12} /> Written answer {candidate.id}
        <span className="muted"> · {look.label}{selected ? ' · selected' : ''}</span>
      </summary>
      <div className="collapse-body">
        {!candidate.transport_complete && (
          <p className="muted small">The response stopped before a finished written answer, so it was kept as partial work and continued rather than reviewed.</p>
        )}
        {open && call && <StreamBody stream={stream} role="solver" />}
        <p className="small">
          <button type="button" className="link-btn" onClick={() => onArtifact(candidate.artifact)}>Open the saved answer</button>
          {call && <> or the <button type="button" className="link-btn" onClick={() => onArtifact(call.request)}>request the solver received</button></>}
        </p>
      </div>
    </details>
  )
}

function ReviewEntry({ review }: { review: Review }) {
  const clean = review.response?.verdict === 'no_issue_found'
  return (
    <div className={'audit' + (clean ? ' audit-complete' : '')}>
      <p className="detail-label">Review {review.id} of {review.candidate}</p>
      {review.response ? <ReviewBody response={review.response} /> : (
        <p className="muted small">No usable review: {review.protocol_error}. A format failure is not a mathematical objection.</p>
      )}
    </div>
  )
}

function LiveCall({ id, role, file, ours }: { id: string; role: string; file: string; ours: boolean }) {
  const stream = useLiveStream('proof', id, file)
  const app = useApp()
  return (
    <div className="live" id="live" aria-live="off">
      <p className="live-head"><Square variant="running" size={14} /> {ROLES[role]?.active || role}</p>
      {ours && <ActivityLog items={app.snapshot.activity} />}
      <StreamBody stream={stream} role={role} live />
    </div>
  )
}

function CallList({ calls, onArtifact }: { calls: Call[]; onArtifact: (name: string) => void }) {
  return (
    <div className="calls-wrap">
      <table className="calls">
        <thead><tr><th>Role</th><th>Result</th><th>Input</th><th>Output</th><th>Time</th><th /></tr></thead>
        <tbody>
          {calls.map((call) => (
            <tr key={call.stream}>
              <td>{ROLES[call.role]?.name || call.role}</td>
              <td>{call.status === 'truncated' ? 'hit its output limit' : call.status}</td>
              <td className="num" title={call.context?.method}>{call.context ? count(call.context.input_tokens) : ''}</td>
              <td className="num">{count(call.charged_tokens ?? call.reserved_tokens)}{call.charged_tokens === undefined ? ' reserved' : ''}</td>
              <td className="num">{call.seconds !== undefined ? duration(call.seconds) : ''}</td>
              <td className="call-links">
                <button type="button" className="link-btn" onClick={() => onArtifact(call.request)}>Request</button>
                <button type="button" className="link-btn" onClick={() => onArtifact(call.stream)}>Output</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function LegacyWork({ data }: { data: ProofDetail }) {
  return (
    <div className="work">
      {data.answer ? (
        <section className="entry candidate">
          <span className="in-margin"><Square variant="complete" size={18} label="Audited candidate" /></span>
          <div>
            <h2 className="candidate-title">Audited candidate from the v0.4 engine</h2>
            <p className="caveat">The earlier engine's whole-proof audit by the same model found no gap. That is a model judgement, not verification.</p>
            <Markdown className="candidate-text">{data.answer}</Markdown>
          </div>
        </section>
      ) : (
        <p className="empty-note">This v0.4 job has no audited candidate.</p>
      )}
      <p className="muted small">Its rounds, claims and objections are in the Report tab, and every saved request and answer is under Files.</p>
    </div>
  )
}

function Overview({ data, onAttempt }: { data: ProofDetail; onAttempt: (index: number) => void }) {
  const b = data.budget
  const settings = data.settings
  return (
    <div className="ledger">
      <section className="aside-section">
        <h3 className="aside-title">Budget</h3>
        <Meter label={data.legacy ? 'Rounds' : 'Attempts'} used={b.rounds.used} limit={b.rounds.limit} />
        <Meter label="Generated tokens" used={b.tokens.used} limit={b.tokens.limit} />
        <Meter label="Time, minutes" used={b.seconds.used / 60} limit={b.seconds.limit / 60} decimals={1} />
        {!data.legacy && settings.max_predict !== undefined && (
          <p className="muted small">
            Each solve reserves {count(settings.max_predict)} tokens{settings.repair_tokens !== undefined && settings.repair_tokens !== settings.max_predict ? `, each repair ${count(settings.repair_tokens)}` : ''}
            {settings.verify_tokens !== undefined ? ` and each review ${count(settings.verify_tokens)}` : ''}; unused reservations are refunded when a call ends.
          </p>
        )}
      </section>
      {!data.legacy && (
        <section className="aside-section">
          <h3 className="aside-title">Candidates {data.candidates.length > 0 && <span className="aside-count">{plural(data.candidates.length, 'answer')}, all kept</span>}</h3>
          {!data.candidates.length && <p className="muted small">No written answer yet. Each attempt saves its answer before it is reviewed.</p>}
          {data.candidates.map((candidate) => {
            const look = candidateLook(candidate)
            const review = [...data.reviews].reverse().find((item) => item.candidate === candidate.id && item.response)
            const issue = review?.response?.issues[0]
            return (
              <article key={candidate.id} className="claim">
                <button type="button" className="claim-head" onClick={() => onAttempt(candidate.attempt)}>
                  <Square variant={look.variant} size={14} label={look.label} />
                  <span className="claim-id">{candidate.id}</span>
                  <span className="claim-status">{look.label}</span>
                  <span className="claim-round">{candidate.id === data.selected_candidate ? 'selected' : `attempt ${candidate.attempt}`}</span>
                </button>
                {issue && <p className="claim-snippet"><Inline limit={150}>{(issue.location ? issue.location + ': ' : '') + issue.evidence}</Inline></p>}
              </article>
            )
          })}
        </section>
      )}
    </div>
  )
}

function ProofReport({ id, tick }: { id: string; tick: number }) {
  const { data, error } = useLoad(() => api.proofReport(id), [id, tick])
  const [which, setWhich] = useState<'report' | 'ledger'>('report')
  return (
    <div className="report">
      {data?.ledger && (
        <div className="segmented" role="group" aria-label="Document">
          <button type="button" aria-pressed={which === 'report'} onClick={() => setWhich('report')}>Debrief</button>
          <button type="button" aria-pressed={which === 'ledger'} onClick={() => setWhich('ledger')}>Full ledger</button>
        </div>
      )}
      <ErrorNote>{error}</ErrorNote>
      {!data && !error && <Loading />}
      {data && (data[which] ? <Markdown>{data[which]}</Markdown> : <p className="empty-note">The harness has not written this file yet.</p>)}
    </div>
  )
}

function ProofSources({ id, goal }: { id: string; goal: string }) {
  const { data, error } = useLoad(() => api.proofSources(id), [id])
  return (
    <div className="sources">
      <section>
        <h3 className="section-title">Original request</h3>
        <pre className="plain">{goal}</pre>
      </section>
      <ErrorNote>{error}</ErrorNote>
      {!data && !error && <Loading />}
      {data && !data.sources.length && <p className="empty-note">No files were pinned. The request above is the whole input.</p>}
      {data?.sources.map((source) => (
        <section key={source.path}>
          <h3 className="section-title">{source.path}</h3>
          <p className="muted small">Snapshot saved when the job started{source.sha256 ? `, SHA-256 ${source.sha256.slice(0, 16)}…` : ''}. Later edits to the file do not change it.</p>
          <SourceLines content={source.content} />
        </section>
      ))}
    </div>
  )
}

export function Files({ names, onOpen }: { names: { name: string; size: number }[]; onOpen: (name: string) => void }) {
  const describe = (name: string) => {
    const kind = name.replace(/^\d+-/, '').replace(/\.(md|jsonl)$/, '')
    if (kind.endsWith('-request')) return `What the ${kind.replace('-request', '')} received`
    if (name.endsWith('.jsonl') || kind.endsWith('-stream')) return `${kind.replace('-stream', '')} output as streamed`
    if (kind === 'proof-candidate') return 'Written answer, kept unchanged'
    if (kind === 'candidate') return 'Solver attempt'
    if (kind === 'checkpoint') return 'Checkpoint from a truncated attempt'
    if (kind === 'tool-result') return 'Tool result'
    if (kind.startsWith('evidence-')) return 'Saved evidence ' + kind.replace('evidence-', '')
    return kind
  }
  if (!names.length) return <p className="empty-note">No files saved yet.</p>
  return (
    <div className="files">
      <p className="muted small">Every request payload, streamed answer and intermediate result is saved, so you can see exactly what the model received and wrote.</p>
      <ul className="file-list">
        {names.map((file) => (
          <li key={file.name}>
            <button type="button" className="file-row" onClick={() => onOpen(file.name)}>
              <span className="file-name">{file.name}</span>
              <span className="file-desc">{describe(file.name)}</span>
              <span className="file-size">{bytes(file.size)}</span>
            </button>
          </li>
        ))}
      </ul>
      <p className="muted small">{plural(names.length, 'file')} in this job's artifacts folder.</p>
    </div>
  )
}
