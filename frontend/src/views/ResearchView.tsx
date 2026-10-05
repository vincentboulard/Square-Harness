import { useCallback, useMemo, useState, type ReactElement } from 'react'
import { api, type Evidence, type ResearchDetail } from '../api'
import { Collapse, ErrorNote, Loading, Meter, SourceLines, Tabs, useLoad } from '../components/common'
import { PauseIcon, PlayIcon } from '../components/Icons'
import { Markdown } from '../components/Markdown'
import { researchLook, researchResumable, Square, type Variant } from '../components/Square'
import { ago, count, firstLine } from '../format'
import { isPanelHidden } from '../panels'
import { go } from '../router'
import { capacityFull, taskForJob } from '../concurrency'
import { useApp, useTick } from '../store'
import { Files } from './ProofView'
import { ActivityLog, ArtifactViewer, BusyNote, JobAside, ROLES, StreamBody, useJobClass, useLiveStream } from './shared'

const STEPS = ['plan', 'investigate', 'draft', 'review', 'revise', 'done']
// A literature review builds a reading list: code searches and follows citations, the model screens and organises.
const LIST_STEPS = ['scope', 'sweep', 'graph', 'screen', 'organise', 'annotate', 'write', 'done']
const STEP_NAMES: Record<string, string> = {
  plan: 'Plan', investigate: 'Investigate', draft: 'Draft', review: 'Review', revise: 'Revise', done: 'Done',
  scope: 'Scope', sweep: 'Search', graph: 'Citations', screen: 'Screen', organise: 'Organise', annotate: 'Annotate', write: 'Write',
}
type Sources = { id: string; path: string; sha256: string; content: string }[]

export function ResearchView({ id, tab }: { id: string; tab: string | null }) {
  const tick = useTick(id)
  const app = useApp()
  const { data, error } = useLoad(() => api.research(id), [id, tick])
  const sources = useLoad(() => api.researchSources(id), [id])
  const [artifact, setArtifact] = useState<string | null>(null)
  const [cite, setCite] = useState<string | null>(null)
  const active = tab || 'report'
  const jobClass = useJobClass()
  const task = taskForJob(app.snapshot, 'research', id)?.task
  const ours = !!task && task.kind === 'research' && task.target === id && ['starting', 'running', 'pausing'].includes(task.state)

  const openCite = useCallback((ref: string) => {
    setCite(ref)
    if (isPanelHidden('aside') || window.matchMedia('(max-width: 1179px)').matches) go(data?.kind || 'literature', id, 'checks')
  }, [data?.kind, id])

  if (!data) {
    return <div className="sheet paper"><div className="sheet-inner">{error ? <ErrorNote>{error}</ErrorNote> : <Loading />}</div></div>
  }
  const flagged = flaggedCitations(data)
  const issues = [...new Set([...data.citation_issues, ...data.review_context_issues, ...data.warnings])]
  return (
    <div className={jobClass}>
      <div className="job-main sheet paper">
        <div className="sheet-inner">
          <ResearchHeader data={data} ours={ours} pausing={task?.state === 'pausing'} />
          <Tabs active={active} onChange={(next) => go(data.kind, id, next === 'report' ? null : next)} tabs={[
            { id: 'report', label: 'Report' },
            { id: 'checks', label: 'Checks', badge: issues.length || undefined, narrowOnly: true },
            { id: 'review', label: 'Model review', hidden: !data.review },
            { id: 'notes', label: 'Plan and notes' },
            { id: 'evidence', label: 'Evidence', badge: data.evidence.length || undefined },
            { id: 'sources', label: 'Sources', badge: data.sources.length || undefined },
            { id: 'files', label: 'Files', badge: data.artifacts.length || undefined },
          ]} />
          {active === 'report' && <Report data={data} flagged={flagged} issues={issues} onCite={openCite} />}
          {active === 'checks' && <div className="ledger-tab"><Checks data={data} sources={sources.data?.sources || null} issues={issues} cite={cite} onCite={setCite} /></div>}
          {active === 'review' && <Markdown>{data.review}</Markdown>}
          {active === 'notes' && <Notes data={data} />}
          {active === 'evidence' && <EvidenceList evidence={data.evidence} />}
          {active === 'sources' && <SourcesTab data={data} sources={sources.data?.sources || null} />}
          {active === 'files' && <Files names={data.artifacts} onOpen={setArtifact} />}
        </div>
      </div>
      <JobAside label="Checks">
        <Checks data={data} sources={sources.data?.sources || null} issues={issues} cite={cite} onCite={setCite} />
      </JobAside>
      {artifact && <ArtifactViewer job="research" id={id} name={artifact} onClose={() => setArtifact(null)} />}
    </div>
  )
}

export function flaggedCitations(data: ResearchDetail): Set<string> {
  const refs = new Set<string>()
  for (const issue of [...data.citation_issues, ...data.review_context_issues]) {
    const match = issue.match(/\[([^\]]+)\]/)
    if (match) refs.add(match[1])
  }
  return refs
}

function ResearchHeader({ data, ours, pausing }: { data: ResearchDetail; ours: boolean; pausing: boolean }) {
  const look = researchLook(data.status, data.running)
  const app = useApp()
  const [error, setError] = useState('')
  const busy = capacityFull(app.snapshot, app.status)
  const ownTask = taskForJob(app.snapshot, 'research', data.id)?.task
  const queued = app.snapshot.queue.some((item) => item.target === data.id)
  const act = (action: Promise<unknown>) => { setError(''); action.catch((reason: Error) => setError(reason.message)) }
  const steps = data.pipeline ? LIST_STEPS : STEPS
  const current = steps.indexOf(data.phase === 'plan' && data.pipeline ? 'scope' : data.phase)
  return (
    <header className="job-head">
      <div className="job-status">
        <Square variant={look.variant} size={18} label={look.label} />
        <span className="job-status-label">{look.label}</span>
        <span className="job-stop">{data.kind === 'referee' ? 'Review' : data.pipeline ? 'Reading list' : 'Literature report'}</span>
      </div>
      <div className="job-goal"><Markdown>{data.goal}</Markdown></div>
      <ol className="phases phases-wide" aria-label="Workflow">
        {steps.map((step, index) => {
          const variant: Variant = index < current || data.phase === 'done' ? 'complete' : index === current ? (data.running ? 'running' : 'paused') : 'ready'
          return <li key={step} className={'phase phase-' + variant}><Square variant={variant} size={10} />{STEP_NAMES[step]}</li>
        })}
      </ol>
      <div className="job-facts">
        {data.sources.map((source) => <span key={source.id} className="fact-file">{source.id} {source.path}</span>)}
        <span>{String(data.settings.model)}</span>
        <span>{data.settings.online ? 'Online' : 'Offline'}</span>
        <span>Started {ago(data.created_at)}</span>
      </div>
      {data.stop_reason && !data.running && <p className="muted small">{data.stop_reason}</p>}
      <div className="job-actions">
        {ours ? (
          <button type="button" className="btn" disabled={pausing} onClick={() => act(api.pause(ownTask!.id))}><PauseIcon size={16} /> {pausing ? 'Pausing…' : 'Pause'}</button>
        ) : researchResumable(data.status) ? (
          <button type="button" className="btn btn-primary" disabled={queued} onClick={() => act(api.resumeResearch(data.id, true))}>
            <PlayIcon size={16} /> {queued ? 'Waiting in the queue' : busy ? 'Queue resume' : 'Resume with the remaining budget'}
          </button>
        ) : null}
        {data.running && !ours && <span className="muted">Running in another process, such as a terminal.</span>}
      </div>
      {!ours && busy && researchResumable(data.status) && !queued && <BusyNote queue />}
      <ErrorNote>{error}</ErrorNote>
    </header>
  )
}

function Report({ data, flagged, issues, onCite }: { data: ResearchDetail; flagged: Set<string>; issues: string[]; onCite: (ref: string) => void }) {
  return (
    <div className="report">
      {data.running && data.live && <LiveResearch id={data.id} role={data.live.role} file={data.live.file} />}
      {issues.length > 0 && (
        <div className="checks-note">
          <p className="objection-label">{data.pipeline ? 'The self-check found gaps' : 'The controller could not confirm every citation'}</p>
          <ul>{issues.slice(0, 6).map((issue) => <li key={issue}>{issue}</li>)}</ul>
          {issues.length > 6 && <p className="small">{issues.length - 6} more in Checks.</p>}
        </div>
      )}
      {data.draft ? (
        <>
          <p className="caveat">{data.pipeline
            ? 'Every entry is a record returned by Crossref, arXiv, zbMATH or Semantic Scholar, with its identifier as returned. The model chose, grouped and annotated them from titles and abstracts.'
            : 'Model draft. Citations link to the passages the harness recorded; red ones cite lines that were never read.'}</p>
          <Markdown className="report-text" onCite={onCite} flagged={flagged}>{data.draft}</Markdown>
        </>
      ) : (
        <p className="empty-note">No draft yet. {data.running ? (data.pipeline ? 'The list is written once the candidates are screened and organised.' : 'The draft follows the investigation.') : 'The plan and working notes are under Plan and notes.'}</p>
      )}
    </div>
  )
}

function LiveResearch({ id, role, file }: { id: string; role: string; file: string }) {
  const stream = useLiveStream('research', id, file)
  const app = useApp()
  const task = taskForJob(app.snapshot, 'research', id)?.task
  const ours = !!task && task.kind === 'research' && task.target === id
  return (
    <div className="live" id="live">
      <p className="live-head"><Square variant="running" size={14} /> {ROLES[role]?.active || role}</p>
      {ours && <ActivityLog items={taskForJob(app.snapshot, 'research', id)?.activity || []} />}
      <StreamBody stream={stream} role={role} live />
    </div>
  )
}

function Checks({ data, sources, issues, cite, onCite }: {
  data: ResearchDetail; sources: Sources | null; issues: string[]; cite: string | null; onCite: (ref: string | null) => void
}) {
  const b = data.budget
  return (
    <div className="ledger">
      {cite && <Passage refText={cite} data={data} sources={sources} onClose={() => onCite(null)} />}
      <section className="aside-section">
        <h3 className="aside-title">Citation checks</h3>
        {issues.length ? (
          <ul className="issue-list">{issues.map((issue) => <li key={issue}>{issue}</li>)}</ul>
        ) : (
          <p className="muted small">{data.draft ? 'Every citation points to lines the harness recorded as read.' : 'Checks run when the draft is written.'}</p>
        )}
      </section>
      {data.sources.length > 0 && (
        <section className="aside-section">
          <h3 className="aside-title">Manuscript coverage</h3>
          {data.sources.map((source) => (
            <Coverage key={source.id} id={source.id} path={source.path} lines={source.lines} ranges={data.manuscript_ranges[source.id] || []} />
          ))}
        </section>
      )}
      <section className="aside-section">
        <h3 className="aside-title">Budget</h3>
        <Meter label="Generated tokens" used={b.tokens.used} limit={b.tokens.limit} />
        <Meter label="Input tokens" used={b.input_tokens.used} limit={b.input_tokens.limit} />
        <Meter label="Time, minutes" used={b.seconds.used / 60} limit={b.seconds.limit / 60} decimals={1} />
        <Meter label="Rounds" used={b.rounds.used} limit={b.rounds.limit} />
        <Meter label="Web requests" used={b.requests.used} limit={b.requests.limit} />
        <Meter label="Evidence characters" used={b.chars.used} limit={b.chars.limit} />
      </section>
    </div>
  )
}

export function Coverage({ id, path, lines, ranges }: { id: string; path: string; lines: number; ranges: [number, number][] }) {
  const read = new Set<number>()
  ranges.forEach(([a, b]) => { for (let n = a; n <= Math.min(b, lines); n++) read.add(n) })
  return (
    <div className="coverage">
      <div className="coverage-text"><span>{id} {path}</span><span className="muted">{count(read.size)} of {count(lines)} lines read</span></div>
      <svg className="coverage-strip" viewBox={`0 0 ${Math.max(lines, 1)} 1`} preserveAspectRatio="none" role="img"
        aria-label={`${read.size} of ${lines} lines read`}>
        <rect x="0" y="0" width={Math.max(lines, 1)} height="1" className="coverage-track" />
        {ranges.map(([a, b], index) => <rect key={index} x={a - 1} y="0" width={Math.max(1, Math.min(b, lines) - a + 1)} height="1" className="coverage-read" />)}
      </svg>
    </div>
  )
}

export function Passage({ refText, data, sources, onClose }: { refText: string; data: ResearchDetail; sources: Sources | null; onClose: () => void }) {
  const [source, range] = refText.split(':L')
  const [start, end] = range ? range.split(/-L?/).map(Number) : [NaN, NaN]
  const last = Number.isFinite(end) ? end : start
  const manuscript = data.sources.find((item) => item.id === source)
  const readRanges = data.manuscript_ranges[source] || []
  const unread = Number.isFinite(start) ? Array.from({ length: Math.max(0, last - start + 1) }, (_, i) => start + i)
    .filter((n) => !readRanges.some(([a, b]) => n >= a && n <= b)) : []
  const passages = useMemo(() => findPassages(data.evidence, source, start, last), [data.evidence, source, start, last])
  return (
    <section className="aside-section passage">
      <div className="passage-head">
        <h3 className="aside-title">{refText}</h3>
        <button type="button" className="link-btn" onClick={onClose}>Close</button>
      </div>
      {manuscript ? (
        <>
          <p className="muted small">{manuscript.path}</p>
          {unread.length > 0 && <p className="objection small">The model cited {unread.length === 1 ? 'a line' : 'lines'} it never read: {firstLine(unread.join(', '), 80)}.</p>}
          {!sources && <Loading />}
          {sources && (() => {
            const text = sources.find((item) => item.id === source)?.content || ''
            const all = text.split('\n')
            const from = Number.isFinite(start) ? Math.max(1, start) : 1
            const to = Number.isFinite(start) ? Math.min(all.length, last) : Math.min(all.length, 40)
            return <SourceLines content={all.slice(from - 1, to).join('\n')} start={from} highlight={readRanges} />
          })()}
        </>
      ) : passages.length ? (
        passages.map((passage, index) => (
          <div key={index} className="passage-quote">
            <p className="muted small">{passage.label}</p>
            <pre className="source">{passage.text}</pre>
          </div>
        ))
      ) : (
        <p className="objection small">No saved passage covers this citation. It may cite search metadata only, or lines that were never read.</p>
      )}
    </section>
  )
}

function findPassages(evidence: Evidence[], source: string, start: number, end: number) {
  const found: { label: string; text: string }[] = []
  for (const item of evidence) {
    let value: Record<string, unknown>
    try { value = JSON.parse(item.result) } catch { continue }
    if (value.document_id !== source) continue
    const a = value.start_line as number | undefined
    const b = value.end_line as number | undefined
    if (typeof value.passage === 'string' && (!Number.isFinite(start) || (a !== undefined && b !== undefined && a <= end && b >= start))) {
      found.push({ label: `${item.id}: ${item.tool}, lines ${a}–${b}`, text: value.passage })
    }
    for (const match of (value.matches as { line: number; passage: string }[] | undefined) || []) {
      if (!Number.isFinite(start) || (match.line >= start && match.line <= end)) found.push({ label: `${item.id}: search hit, line ${match.line}`, text: match.passage })
    }
  }
  return found
}

function Notes({ data }: { data: ResearchDetail }) {
  return (
    <div className="notes">
      <section>
        <h3 className="section-title">{data.pipeline ? 'Scope' : 'Plan'}</h3>
        {data.plan ? <Markdown>{data.plan}</Markdown> : <p className="empty-note">No plan yet.</p>}
      </section>
      {data.notes.filter((note) => note.text.trim()).map((note, index) => (
        <section key={index}>
          <h3 className="section-title">{note.title || `Working note, round ${note.round}`}{note.complete ? '' : ', unfinished'}</h3>
          <Markdown>{note.text}</Markdown>
        </section>
      ))}
      {!data.notes.some((note) => note.text.trim()) && <p className="muted small">The investigation rounds left no written notes; their tool calls are under Evidence.</p>}
    </div>
  )
}

function EvidenceList({ evidence }: { evidence: Evidence[] }) {
  if (!evidence.length) return <p className="empty-note">No source was read yet. Evidence appears as the investigation calls its tools.</p>
  return (
    <div className="evidence">
      {evidence.map((item) => {
        let body: ReactElement
        try {
          const value = JSON.parse(item.result)
          if (Array.isArray(value.lines)) body = <SourceLines content={value.lines.map((line: { text: string }) => line.text).join('\n')} start={value.start_line} />
          else if (typeof value.passage === 'string') body = <pre className="source">{value.passage}</pre>
          else if (value.error) body = <p className="objection small">{String(value.error)}</p>
          else body = <pre className="plain">{JSON.stringify(value, null, 2)}</pre>
        } catch {
          body = <pre className="plain">{item.result}</pre>
        }
        return (
          <Collapse key={item.id} summary={<><strong>{item.id}</strong> <span className="tool-line">{item.tool}</span> <span className="muted">{firstLine(JSON.stringify(item.arguments), 90)}</span></>}>
            {body}
          </Collapse>
        )
      })}
    </div>
  )
}

export function SourcesTab({ data, sources }: { data: ResearchDetail; sources: Sources | null }) {
  if (!data.sources.length) return <p className="empty-note">No manuscript was pinned for this job.</p>
  if (!sources) return <Loading />
  return (
    <div className="sources">
      {sources.map((source) => (
        <section key={source.id}>
          <h3 className="section-title">{source.id} {source.path}</h3>
          <p className="muted small">Snapshot saved when the job started. Lines the harness read are marked in the margin.</p>
          <SourceLines content={source.content} highlight={data.manuscript_ranges[source.id] || []} />
        </section>
      ))}
    </div>
  )
}
