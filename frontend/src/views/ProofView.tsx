import { useEffect, useMemo, useState } from 'react'
import { api, type Audit, type Call, type Claim, type Critique, type ProofDetail, type Round } from '../api'
import { Collapse, ErrorNote, Loading, Meter, SourceLines, Tabs, useLoad } from '../components/common'
import { CheckIcon, PauseIcon, PlayIcon } from '../components/Icons'
import { Inline, Markdown } from '../components/Markdown'
import { claimLook, proofLook, proofResumable, Square, type Variant } from '../components/Square'
import { ago, bytes, count, duration, plural } from '../format'
import { go } from '../router'
import { useApp, useTick } from '../store'
import { ActivityLog, ArtifactViewer, BusyNote, ROLES, StreamBody, useLiveStream } from './shared'

const PHASES = ['plan', 'solve', 'checkpoint', 'critic', 'review', 'audit'] as const
const PHASE_NAMES: Record<string, string> = {
  plan: 'Plan', solve: 'Solve', checkpoint: 'Checkpoint', critic: 'Critique', review: 'Record', audit: 'Audit',
}

export function ProofView({ id, tab }: { id: string; tab: string | null }) {
  const tick = useTick(id)
  const app = useApp()
  const { data, error } = useLoad(() => api.proof(id), [id, tick])
  const [artifact, setArtifact] = useState<string | null>(null)
  const [focus, setFocus] = useState<string | null>(null)
  const active = tab || 'work'
  const task = app.snapshot.task
  const ours = !!task && task.kind === 'proof' && task.target === id && ['starting', 'running', 'pausing'].includes(task.state)

  const focusClaim = (claimId: string | null) => {
    setFocus(claimId)
    if (!claimId) return
    if (window.matchMedia('(max-width: 1179px)').matches) go('prove', id, 'ledger')
    window.setTimeout(() => document.getElementById('claim-' + claimId)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }), 60)
  }

  if (!data) {
    return <div className="sheet paper"><div className="sheet-inner">{error ? <ErrorNote>{error}</ErrorNote> : <Loading />}</div></div>
  }
  const counts = data.claims.reduce<Record<string, number>>((acc, claim) => ({ ...acc, [claim.status]: (acc[claim.status] || 0) + 1 }), {})
  return (
    <div className="job">
      <div className="job-main sheet paper">
        <div className="sheet-inner">
          <ProofHeader data={data} ours={ours} pausing={task?.state === 'pausing'} />
          <Tabs active={active} onChange={(next) => go('prove', id, next === 'work' ? null : next)} tabs={[
            { id: 'work', label: 'Work', badge: data.rounds.length + (data.pending ? 1 : 0) || undefined },
            { id: 'ledger', label: 'Ledger', badge: data.claims.length || undefined, narrowOnly: true },
            { id: 'report', label: 'Report' },
            { id: 'sources', label: 'Sources', badge: data.sources.length || undefined },
            { id: 'files', label: 'Files', badge: data.artifacts.length || undefined },
          ]} />
          {active === 'work' && <Work data={data} ours={ours} onArtifact={setArtifact} onClaim={focusClaim} />}
          {active === 'ledger' && <div className="ledger-tab"><Ledger data={data} counts={counts} focus={focus} onFocus={focusClaim} onArtifact={setArtifact} /></div>}
          {active === 'report' && <ProofReport id={id} tick={tick} />}
          {active === 'sources' && <ProofSources id={id} goal={data.goal} />}
          {active === 'files' && <Files names={data.artifacts} onOpen={setArtifact} />}
        </div>
      </div>
      <aside className="job-aside" aria-label="Ledger">
        <Ledger data={data} counts={counts} focus={focus} onFocus={focusClaim} onArtifact={setArtifact} />
      </aside>
      {artifact && <ArtifactViewer job="proof" id={id} name={artifact} onClose={() => setArtifact(null)} />}
    </div>
  )
}

function ProofHeader({ data, ours, pausing }: { data: ProofDetail; ours: boolean; pausing: boolean }) {
  const look = proofLook(data.status, data.running)
  const [error, setError] = useState('')
  const [expanded, setExpanded] = useState(false)
  const app = useApp()
  const busy = !!app.snapshot.task && ['starting', 'running', 'pausing'].includes(app.snapshot.task.state)
  const long = data.goal.length > 420
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
        <span>{String(data.settings.model)}</span>
        <span>Context {count(data.settings.ctx)}</span>
        <span>Thinking {data.settings.think ? 'on' : 'off'}</span>
        <span>Literature tools {data.settings.allow_literature ? 'on' : 'off'}</span>
        <span>Started {ago(data.created_at)}</span>
      </div>
      {data.recovery_notice && <p className="warning-note">{data.recovery_notice}</p>}
      <div className="job-actions">
        {ours ? (
          <button type="button" className="btn" disabled={pausing} onClick={() => act(api.pause())}>
            <PauseIcon size={16} /> {pausing ? 'Pausing…' : 'Pause'}
          </button>
        ) : proofResumable(data.status) ? (
          <button type="button" className="btn btn-primary" disabled={busy} onClick={() => act(api.resumeProof(data.id))}>
            <PlayIcon size={16} /> Resume with the remaining budget
          </button>
        ) : null}
        {data.running && !ours && <span className="muted">Running in another process, such as a terminal.</span>}
      </div>
      {!ours && busy && proofResumable(data.status) && <BusyNote />}
      <ErrorNote>{error}</ErrorNote>
    </header>
  )
}

function Work({ data, ours, onArtifact, onClaim }: { data: ProofDetail; ours: boolean; onArtifact: (name: string) => void; onClaim: (id: string) => void }) {
  useEffect(() => {
    if (data.running) document.getElementById('live')?.scrollIntoView({ block: 'center' })
    // Only when the page opens on a running job.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  return (
    <div className="work">
      {data.candidate && <CompleteCandidate text={data.candidate} audit={data.final_audit} />}
      {!data.rounds.length && !data.pending && (
        <p className="empty-note">No round has started yet. Resume the job to begin.</p>
      )}
      {data.rounds.map((round) => (
        <RoundEntry key={round.index} round={round} data={data} onArtifact={onArtifact} onClaim={onClaim} />
      ))}
      {data.pending && <RoundEntry round={data.pending} data={data} pending ours={ours} onArtifact={onArtifact} onClaim={onClaim} />}
    </div>
  )
}

function CompleteCandidate({ text, audit }: { text: string; audit: Audit | null }) {
  return (
    <section className="entry candidate">
      <span className="in-margin"><Square variant="complete" size={18} label="Complete candidate" /></span>
      <div>
        <h2 className="candidate-title">Complete candidate</h2>
        <p className="caveat">A whole-proof audit by the same model found no gap. That is a model judgement, not verification: check every step yourself.</p>
        <Markdown className="candidate-text">{text}</Markdown>
        <p className="tombstone" aria-hidden="true">∎</p>
        {audit?.explanation && (
          <Collapse summary="Why the auditor accepted it"><Markdown>{audit.explanation}</Markdown></Collapse>
        )}
      </div>
    </section>
  )
}

function phasesFor(round: Round, pending: boolean, audited: boolean, running: boolean) {
  const visited = PHASES.filter((phase) =>
    phase === 'plan' ? round.fresh || !!round.plan
      : phase === 'checkpoint' ? !!round.solver_truncated || !!round.raw_draft
      : phase === 'audit' ? audited || round.phase === 'audit'
      : true)
  const current = pending ? PHASES.indexOf(round.phase as typeof PHASES[number]) : -1
  return visited.map((phase) => {
    const index = PHASES.indexOf(phase)
    const variant: Variant = !pending ? 'complete' : index < current ? 'complete' : index === current ? (running ? 'running' : 'paused') : 'ready'
    return { phase, variant }
  })
}

function RoundEntry({ round, data, pending = false, ours = false, onArtifact, onClaim }: {
  round: Round; data: ProofDetail; pending?: boolean; ours?: boolean; onArtifact: (name: string) => void; onClaim: (id: string) => void
}) {
  const calls = data.calls.filter((call) => call.round === round.index)
  const claims = (round.claim_ids || []).map((claimId) => data.claims.find((claim) => claim.id === claimId)).filter((claim): claim is Claim => !!claim)
  const audit = data.final_audit && data.final_audit.round === round.index ? data.final_audit : null
  const phases = phasesFor(round, pending, !!audit, data.running)
  const interrupted = pending && !data.running ? [...calls].reverse().find((call) => call.status === 'interrupted') : undefined
  const heading = round.plan ? 'new approach' : round.assembling ? 'assembling the proof' : round.fresh ? 'fresh start' : ''
  return (
    <section className={'entry round' + (pending ? ' round-pending' : '')}>
      <span className="in-margin round-number">{round.index}</span>
      <div className="round-body">
        <div className="round-head">
          <h2>Round {round.index}{heading && <span className="round-kind">, {heading}</span>}</h2>
          <ol className="phases" aria-label="Phases">
            {phases.map(({ phase, variant }) => (
              <li key={phase} className={'phase phase-' + variant}><Square variant={variant} size={10} />{PHASE_NAMES[phase]}</li>
            ))}
          </ol>
        </div>
        {round.plan && (
          <div className="plan">
            <Markdown>{`**${round.plan.approach}**`}</Markdown>
            <Markdown className="muted-prose">{round.plan.difference}</Markdown>
          </div>
        )}
        <Collapse summary="Task for this round" open={pending}>
          <Markdown>{round.task}</Markdown>
        </Collapse>
        {pending && data.live && data.running && <LiveCall id={data.id} role={data.live.role} file={data.live.file} ours={ours} />}
        {pending && !data.running && <p className="muted small">This round stopped in the {PHASE_NAMES[round.phase] || round.phase} phase. Resuming continues from its last saved checkpoint.</p>}
        {interrupted && <PartialCall id={data.id} call={interrupted} />}
        {round.draft && <Attempt id={data.id} name={round.draft} raw={round.raw_draft} onArtifact={onArtifact} />}
        {round.critique && <CritiqueView critique={round.critique} />}
        {claims.length > 0 && (
          <div className="recorded">
            {claims.map((claim) => {
              const look = claimLook(claim.status)
              return (
                <button key={claim.id} type="button" className={'claim-chip claim-chip-' + claim.status} onClick={() => onClaim(claim.id)} title={look.label}>
                  <Square variant={look.variant} size={12} /> <span className="claim-chip-id">{claim.id}</span>
                  <Inline className="claim-chip-text" limit={70}>{claim.statement || claim.objection || 'No claim extracted'}</Inline>
                </button>
              )
            })}
          </div>
        )}
        {round.strategy_summary && <p className="summary"><span className="summary-label">Where it got to</span> <Inline>{round.strategy_summary}</Inline></p>}
        {audit && <AuditView audit={audit} />}
        {calls.length > 0 && (
          <Collapse summary={<>Model calls <span className="muted">{calls.length}</span></>}>
            <CallList calls={calls} onArtifact={onArtifact} />
          </Collapse>
        )}
      </div>
    </section>
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

function PartialCall({ id, call }: { id: string; call: Call }) {
  const [open, setOpen] = useState(false)
  const stream = useLiveStream('proof', id, open ? call.stream : null)
  return (
    <details className="collapse" onToggle={(event) => setOpen((event.currentTarget as HTMLDetailsElement).open)}>
      <summary>Partial output of the interrupted {ROLES[call.role]?.name.toLowerCase() || call.role} call</summary>
      <div className="collapse-body">
        <p className="muted small">Kept as unreviewed material. Its token reservation stays charged, as in the terminal.</p>
        {open && <StreamBody stream={stream} role={call.role} />}
      </div>
    </details>
  )
}

function Attempt({ id, name, raw, onArtifact }: { id: string; name: string; raw?: string; onArtifact: (name: string) => void }) {
  const [value, setValue] = useState<{ text: string; thinking: string } | null>(null)
  const [error, setError] = useState('')
  const load = () => {
    if (value) return
    api.proofArtifact(id, name).then((artifact) => {
      const json = artifact.json as { text?: string; thinking?: string } | undefined
      setValue({ text: json?.text || '', thinking: json?.thinking || '' })
    }, (reason: Error) => setError(reason.message))
  }
  return (
    <details className="collapse attempt" onToggle={(event) => (event.currentTarget as HTMLDetailsElement).open && load()}>
      <summary>{raw ? 'Checkpoint written from a truncated attempt' : 'Written attempt'}</summary>
      <div className="collapse-body">
        <ErrorNote>{error}</ErrorNote>
        {!value && !error && <Loading />}
        {value && (
          <StreamBody stream={{ text: value.text || '*The attempt contained no written answer; only thinking was produced.*', thinking: value.thinking, tool_calls: [], done: true, stats: {} }} role="solver" />
        )}
        <p className="small">
          <button type="button" className="link-btn" onClick={() => onArtifact(name)}>Open the saved file</button>
          {raw && <> or the <button type="button" className="link-btn" onClick={() => onArtifact(raw)}>original truncated attempt</button></>}
        </p>
      </div>
    </details>
  )
}

function CritiqueView({ critique }: { critique: Critique }) {
  const clean = !critique.first_invalid_step.trim() && !critique.missing_work.trim()
  return (
    <div className="critique">
      {critique.valid_steps.map((step, index) => (
        <div key={index} className="valid-step"><CheckIcon size={16} /><Markdown>{step}</Markdown></div>
      ))}
      {critique.first_invalid_step.trim() && (
        <div className="objection">
          <p className="objection-label">First unsupported step</p>
          <Markdown>{critique.first_invalid_step}</Markdown>
          {critique.reason.trim() && <Markdown className="objection-reason">{critique.reason}</Markdown>}
        </div>
      )}
      {critique.missing_work.trim() && (
        <div className="missing">
          <p className="missing-label">Still missing</p>
          <Markdown>{critique.missing_work}</Markdown>
        </div>
      )}
      {clean && critique.complete_candidate && <p className="valid-note">The critic found no gap in this attempt.</p>}
    </div>
  )
}

function AuditView({ audit }: { audit: Audit }) {
  const complete = audit.verdict === 'complete'
  return (
    <div className={'audit' + (complete ? ' audit-complete' : '')}>
      <p className="audit-verdict">Whole-proof audit: {complete ? 'no gap found' : audit.verdict === 'gap' ? 'gap found' : 'uncertain'}</p>
      {audit.explanation && <Markdown>{audit.explanation}</Markdown>}
      {audit.objection && <div className="objection"><Markdown>{audit.objection}</Markdown></div>}
    </div>
  )
}

function CallList({ calls, onArtifact }: { calls: Call[]; onArtifact: (name: string) => void }) {
  return (
    <table className="calls">
      <thead><tr><th>Role</th><th>Result</th><th>Tokens</th><th>Time</th><th /></tr></thead>
      <tbody>
        {calls.map((call) => (
          <tr key={call.stream}>
            <td>{ROLES[call.role]?.name || call.role}</td>
            <td>{call.status === 'truncated' ? 'hit its output limit' : call.status}</td>
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
  )
}

function Ledger({ data, counts, focus, onFocus, onArtifact }: {
  data: ProofDetail; counts: Record<string, number>; focus: string | null; onFocus: (id: string | null) => void; onArtifact: (name: string) => void
}) {
  const b = data.budget
  const tally = ['reviewed', 'gap', 'refuted', 'uncertain'].filter((status) => counts[status])
    .map((status) => `${counts[status]} ${claimLook(status).label.toLowerCase()}`).join(', ')
  return (
    <div className="ledger">
      <section className="aside-section">
        <h3 className="aside-title">Budget</h3>
        <Meter label="Rounds" used={b.rounds.used} limit={b.rounds.limit} />
        <Meter label="Generated tokens" used={b.tokens.used} limit={b.tokens.limit} />
        <Meter label="Time, minutes" used={b.seconds.used / 60} limit={b.seconds.limit / 60} decimals={1} />
      </section>
      {data.status !== 'candidate_complete' && data.next_task && (
        <section className="aside-section">
          <h3 className="aside-title">Next task</h3>
          <Markdown className="aside-prose">{data.next_task}</Markdown>
        </section>
      )}
      <section className="aside-section">
        <h3 className="aside-title">Claims {tally && <span className="aside-count">{tally}</span>}</h3>
        {!data.claims.length && <p className="muted small">No claims recorded yet. The recorder adds them after each critique.</p>}
        {data.claims.map((claim) => (
          <ClaimEntry key={claim.id} claim={claim} claims={data.claims} open={focus === claim.id} onFocus={onFocus} onArtifact={onArtifact} />
        ))}
      </section>
    </div>
  )
}

function ClaimEntry({ claim, claims, open, onFocus, onArtifact }: {
  claim: Claim; claims: Claim[]; open: boolean; onFocus: (id: string | null) => void; onArtifact: (name: string) => void
}) {
  const look = claimLook(claim.status)
  const usedBy = useMemo(() => claims.filter((other) => other.dependencies.includes(claim.id)).map((other) => other.id), [claims, claim.id])
  return (
    <article id={'claim-' + claim.id} className={'claim claim-' + claim.status + (open ? ' claim-open' : '')}>
      <button type="button" className="claim-head" aria-expanded={open} onClick={() => onFocus(open ? null : claim.id)}>
        <Square variant={look.variant} size={14} label={look.label} />
        <span className="claim-id">{claim.id}</span>
        <span className="claim-status">{look.label}</span>
        <span className="claim-round">round {claim.round}</span>
      </button>
      <div className={open ? '' : 'claim-clamp'}>
        <Markdown className="claim-statement">{claim.statement || '*No claim was extracted.*'}</Markdown>
      </div>
      {!open && claim.objection && <p className="claim-snippet"><Inline limit={150}>{claim.objection}</Inline></p>}
      {open && (
        <div className="claim-detail">
          {claim.assumptions.length > 0 && (
            <div><p className="detail-label">Assumptions</p><ul>{claim.assumptions.map((item, index) => <li key={index}><Markdown>{item}</Markdown></li>)}</ul></div>
          )}
          <ClaimLinks label="Depends on" ids={claim.dependencies} claims={claims} onFocus={onFocus} />
          <ClaimLinks label="Used by" ids={usedBy} claims={claims} onFocus={onFocus} />
          {claim.argument && <div><p className="detail-label">Argument</p><Markdown>{claim.argument}</Markdown></div>}
          {claim.objection && <div className="objection"><p className="objection-label">Objection</p><Markdown>{claim.objection}</Markdown></div>}
          {claim.whole_proof_objection && <div className="objection"><p className="objection-label">Whole-proof audit objection</p><Markdown>{claim.whole_proof_objection}</Markdown></div>}
          {claim.evidence && <div><p className="detail-label">Evidence or witness</p><Markdown>{claim.evidence}</Markdown></div>}
          {claim.resolves.length > 0 && (
            <div><p className="detail-label">Answers the objections of {claim.resolves.join(', ')}</p><Markdown>{claim.resolution}</Markdown></div>
          )}
          <p className="small">
            <button type="button" className="link-btn" onClick={() => onArtifact(claim.candidate_artifact)}>Attempt it came from</button>
            {claim.review_artifact && <> and the <button type="button" className="link-btn" onClick={() => onArtifact(claim.review_artifact!)}>recorder's answer</button></>}
          </p>
        </div>
      )}
    </article>
  )
}

function ClaimLinks({ label, ids, claims, onFocus }: { label: string; ids: string[]; claims: Claim[]; onFocus: (id: string) => void }) {
  if (!ids.length) return null
  return (
    <div className="claim-links">
      <span className="detail-label">{label}</span>
      {ids.map((ref) => {
        const target = claims.find((claim) => claim.id === ref)
        return target ? (
          <button key={ref} type="button" className="claim-chip" onClick={() => onFocus(ref)}>
            <Square variant={claimLook(target.status).variant} size={11} /> {ref}
          </button>
        ) : <span key={ref} className="claim-chip claim-chip-missing" title="Not a recorded claim">{ref}</span>
      })}
    </div>
  )
}

function ProofReport({ id, tick }: { id: string; tick: number }) {
  const { data, error } = useLoad(() => api.proofReport(id), [id, tick])
  const [which, setWhich] = useState<'report' | 'ledger'>('report')
  return (
    <div className="report">
      <div className="segmented" role="group" aria-label="Document">
        <button type="button" aria-pressed={which === 'report'} onClick={() => setWhich('report')}>Debrief</button>
        <button type="button" aria-pressed={which === 'ledger'} onClick={() => setWhich('ledger')}>Full ledger</button>
      </div>
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
