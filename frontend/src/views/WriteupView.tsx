import { useMemo, useState, type ReactNode } from 'react'
import { api, type ResearchDetail } from '../api'
import { ErrorNote, Loading, Meter, Tabs, useLoad } from '../components/common'
import { PauseIcon, PlayIcon } from '../components/Icons'
import { Markdown } from '../components/Markdown'
import { researchLook, researchResumable, Square, type Variant } from '../components/Square'
import { ago } from '../format'
import { isPanelHidden } from '../panels'
import { go } from '../router'
import { capacityFull, taskForJob } from '../concurrency'
import { useApp, useTick } from '../store'
import { Files } from './ProofView'
import { flaggedCitations, Passage, SourcesTab } from './ResearchView'
import { ActivityLog, ArtifactViewer, BusyNote, JobAside, ROLES, StreamBody, useJobClass, useLiveStream } from './shared'

const STEPS = ['plan', 'section', 'assemble', 'compile', 'review', 'export'] as const
const STEP_NAMES: Record<string, string> = {
  plan: 'Outline', section: 'Sections', assemble: 'Assemble', compile: 'Compile', repair: 'Repair',
  review: 'Review', export: 'Export', done: 'Done',
}

export function WriteupView({ id, tab }: { id: string; tab: string | null }) {
  const tick = useTick(id)
  const app = useApp()
  const { data, error } = useLoad(() => api.research(id), [id, tick])
  const sources = useLoad(() => api.researchSources(id), [id])
  const [artifact, setArtifact] = useState<string | null>(null)
  const [cite, setCite] = useState<string | null>(null)
  const active = tab || 'document'
  const jobClass = useJobClass()
  const task = taskForJob(app.snapshot, 'research', id)?.task
  const ours = !!task && task.kind === 'research' && task.target === id && ['starting', 'running', 'pausing'].includes(task.state)
  const openCite = (ref: string) => {
    setCite(ref)
    if (isPanelHidden('aside') || window.matchMedia('(max-width: 1179px)').matches) go('writeup', id, 'checks')
  }
  if (!data) {
    return <div className="sheet paper"><div className="sheet-inner">{error ? <ErrorNote>{error}</ErrorNote> : <Loading />}</div></div>
  }
  const issues = [...new Set([...data.citation_issues, ...data.warnings])]
  const aside = <Aside data={data} issues={issues} cite={cite} onCite={setCite} sources={sources.data?.sources || null} />
  return (
    <div className={jobClass}>
      <div className="job-main sheet paper">
        <div className="sheet-inner">
          <Header data={data} ours={ours} pausing={task?.state === 'pausing'} />
          <Tabs active={active} onChange={(next) => go('writeup', id, next === 'document' ? null : next)} tabs={[
            { id: 'document', label: 'Document' },
            { id: 'checks', label: 'Checks', badge: issues.length || undefined, narrowOnly: true },
            { id: 'compile', label: 'Compile', badge: data.compile?.errors.length || undefined },
            { id: 'review', label: 'Model review', hidden: !data.review },
            { id: 'outline', label: 'Outline' },
            { id: 'sources', label: 'Sources', badge: data.sources.length || undefined },
            { id: 'files', label: 'Files', badge: data.artifacts.length || undefined },
          ]} />
          {active === 'document' && <DocumentTab data={data} ours={ours} onCite={openCite} />}
          {active === 'checks' && <div className="ledger-tab">{aside}</div>}
          {active === 'compile' && <CompileTab data={data} onArtifact={setArtifact} />}
          {active === 'review' && <Markdown>{data.review}</Markdown>}
          {active === 'outline' && <OutlineTab data={data} />}
          {active === 'sources' && <SourcesTab data={data} sources={sources.data?.sources || null} />}
          {active === 'files' && <Files names={data.artifacts} onOpen={setArtifact} />}
        </div>
      </div>
      <JobAside label="Checks">{aside}</JobAside>
      {artifact && <ArtifactViewer job="research" id={id} name={artifact} onClose={() => setArtifact(null)} />}
    </div>
  )
}

function Header({ data, ours, pausing }: { data: ResearchDetail; ours: boolean; pausing: boolean }) {
  const app = useApp()
  const look = researchLook(data.status, data.running)
  const [error, setError] = useState('')
  const busy = capacityFull(app.snapshot, app.status)
  const ownTask = taskForJob(app.snapshot, 'research', data.id)?.task
  const queued = app.snapshot.queue.some((item) => item.target === data.id)
  const act = (action: Promise<unknown>) => { setError(''); action.catch((reason: Error) => setError(reason.message)) }
  const phase = data.phase === 'repair' ? 'compile' : data.phase
  const current = STEPS.indexOf(phase as typeof STEPS[number])
  return (
    <header className="job-head">
      <div className="job-status">
        <Square variant={look.variant} size={18} label={look.label} />
        <span className="job-status-label">{look.label}</span>
        <span className="job-stop">Write-up</span>
      </div>
      <div className="job-goal"><Markdown>{data.goal}</Markdown></div>
      <ol className="phases phases-wide" aria-label="Workflow">
        {STEPS.map((step, index) => {
          const variant: Variant = index < current || data.phase === 'done' ? 'complete' : index === current ? (data.running ? 'running' : 'paused') : 'ready'
          return <li key={step} className={'phase phase-' + variant}><Square variant={variant} size={10} />{step === 'compile' && data.phase === 'repair' ? 'Repair' : STEP_NAMES[step]}</li>
        })}
      </ol>
      <div className="job-facts">
        {data.sources.map((source) => <span key={source.id} className="fact-file">{source.id} {source.path}</span>)}
        <span>{String(data.settings.model)}</span>
        <span>Started {ago(data.created_at)}</span>
      </div>
      {(data.outputs?.tex || data.compile?.pdf) && (
        <div className="outputs">
          {data.outputs?.tex && <span className="output-file">Saved as <code>{data.outputs.tex}</code>{data.outputs.pdf && <> and <code>{data.outputs.pdf}</code></>} in your folder</span>}
          {data.compile?.pdf && <a className="btn btn-small" href={api.researchPdfUrl(data.id)} target="_blank" rel="noopener">Open the PDF</a>}
        </div>
      )}
      <div className="job-actions">
        {ours ? (
          <button type="button" className="btn" disabled={pausing} onClick={() => act(api.pause(ownTask!.id))}><PauseIcon size={16} /> {pausing ? 'Pausing…' : 'Pause'}</button>
        ) : researchResumable(data.status) ? (
          <button type="button" className="btn btn-primary" disabled={queued} onClick={() => act(api.resumeResearch(data.id, true))}>
            <PlayIcon size={16} /> {queued ? 'Waiting in the queue' : busy ? 'Queue resume' : 'Resume with the remaining budget'}
          </button>
        ) : null}
      </div>
      {!ours && busy && researchResumable(data.status) && !queued && <BusyNote queue />}
      <ErrorNote>{error}</ErrorNote>
    </header>
  )
}

function DocumentTab({ data, ours, onCite }: { data: ResearchDetail; ours: boolean; onCite: (ref: string) => void }) {
  const flagged = flaggedCitations(data)
  const errorLines = new Set((data.compile?.errors || []).filter((e) => e.file === 'document.tex' && e.line).map((e) => e.line as number))
  return (
    <div className="document">
      {data.running && data.live && <LiveWriteup id={data.id} role={data.live.role} file={data.live.file} ours={ours} />}
      {data.document ? (
        <>
          <p className="caveat">Model draft. Each <code>% src</code> mark opens the note lines a paragraph came from; red ones cite lines that were never read. TODO comments are left for you.</p>
          <TexSource text={data.document} onCite={onCite} flagged={flagged} errorLines={errorLines} />
        </>
      ) : data.sections?.some((s) => s.complete) ? (
        <>
          <p className="caveat">Sections written so far. The full document appears once all sections are done.</p>
          {data.sections.filter((s) => s.complete).map((section, index) => (
            <div key={index}><h3 className="section-title">{section.title}</h3><TexSource text={section.latex} onCite={onCite} flagged={flagged} errorLines={new Set()} /></div>
          ))}
        </>
      ) : (
        <p className="empty-note">No section written yet.{data.running ? (data.phase === 'plan' ? ' The outline comes first.' : ' The first one is being written.') : ''}</p>
      )}
    </div>
  )
}

function LiveWriteup({ id, role, file, ours }: { id: string; role: string; file: string; ours: boolean }) {
  const stream = useLiveStream('research', id, file)
  const app = useApp()
  return (
    <div className="live" id="live">
      <p className="live-head"><Square variant="running" size={14} /> {ROLES[role === 'plan' ? 'outline' : role]?.active || role}</p>
      {ours && <ActivityLog items={taskForJob(app.snapshot, 'research', id)?.activity || []} />}
      {role === 'plan' ? <StreamBody stream={stream} role="outline" live />
        : role === 'fidelity' ? <StreamBody stream={stream} role="fidelity" live />
        : <pre className="source tex live-tex">{stream.text || 'Waiting for the first tokens…'}</pre>}
    </div>
  )
}

// A light LaTeX highlighter: commands, comments, TODOs and the provenance marks.
function highlight(line: string): ReactNode[] {
  const parts: ReactNode[] = []
  const comment = line.search(/(?<!\\)%/)
  const code = comment >= 0 ? line.slice(0, comment) : line
  code.split(/(\\[A-Za-z@]+)/).forEach((piece, index) => {
    if (!piece) return
    parts.push(piece.startsWith('\\') ? <span key={index} className="tex-command">{piece}</span> : piece)
  })
  if (comment >= 0) {
    const text = line.slice(comment)
    parts.push(<span key="comment" className={/%\s*TODO/.test(text) ? 'tex-todo' : 'tex-comment'}>{text}</span>)
  }
  return parts
}

function TexSource({ text, onCite, flagged, errorLines }: { text: string; onCite: (ref: string) => void; flagged: Set<string>; errorLines: Set<number> }) {
  const lines = useMemo(() => text.split('\n'), [text])
  return (
    <pre className="source tex">
      {lines.map((line, index) => {
        const provenance = line.match(/^(\s*)%\s*src:\s*(.*)$/)
        const refs = provenance ? [...provenance[2].matchAll(/\[([^\]]+)\]/g)].map((m) => m[1]) : []
        return (
          <div key={index} className={'source-line' + (errorLines.has(index + 1) ? ' source-error' : '') + (provenance ? ' source-provenance' : '')}>
            <span className="source-number">{index + 1}</span>
            <span className="source-text">
              {provenance ? (
                <>
                  {provenance[1]}<span className="tex-comment">% from </span>
                  {refs.length ? refs.map((ref) => (
                    <button key={ref} type="button" className={'cite' + (flagged.has(ref) ? ' cite-flagged' : '')} onClick={() => onCite(ref)}>{ref}</button>
                  )) : <span className="tex-comment">{provenance[2]}</span>}
                </>
              ) : (line ? highlight(line) : ' ')}
            </span>
          </div>
        )
      })}
    </pre>
  )
}

function CompileTab({ data, onArtifact }: { data: ResearchDetail; onArtifact: (name: string) => void }) {
  const result = data.compile
  if (!result) return <p className="empty-note">Not compiled yet. Compilation follows the sections.</p>
  if (!result.available) return <p className="empty-note">LaTeX (latexmk) is not installed on this computer, so the document was not compiled.</p>
  return (
    <div className="compile">
      <p className={result.ok ? 'valid-note' : 'objection small'}>
        {result.ok ? 'The document compiled without errors.' : `The document did not compile: ${result.errors.length} error${result.errors.length === 1 ? '' : 's'}.`}
        {data.phase === 'done' && !result.ok && ' One repair round was tried; fix the remaining errors in the .tex file.'}
      </p>
      {result.pdf && <p><a className="btn" href={api.researchPdfUrl(data.id)} target="_blank" rel="noopener">Open the PDF</a></p>}
      {result.errors.length > 0 && (
        <ul className="issue-list">
          {result.errors.map((item, index) => (
            <li key={index}>{item.file ? `${item.file}${item.line ? ':' + item.line : ''} ` : ''}{item.message}</li>
          ))}
        </ul>
      )}
      <p className="muted small">Compiled with latexmk in a temporary folder, without shell escape and with TeX file access limited to that folder.</p>
      {result.log && <button type="button" className="link-btn" onClick={() => onArtifact(result.log!)}>Open the full LaTeX log</button>}
    </div>
  )
}

function OutlineTab({ data }: { data: ResearchDetail }) {
  if (!data.outline) return <p className="empty-note">No outline yet.</p>
  return (
    <div className="outline">
      <h3 className="section-title">{data.outline.title}</h3>
      <ol className="outline-list">
        {(data.sections || []).map((section, index) => (
          <li key={index}>
            <strong>{section.title}</strong>{section.truncated && <span className="objection-label"> cut short by the output limit</span>}
            <p className="muted small">{section.goal} From {section.sources.join(', ') || 'no lines'}.</p>
          </li>
        ))}
      </ol>
    </div>
  )
}

type Sources = { id: string; path: string; sha256: string; content: string }[]

function Aside({ data, issues, cite, onCite, sources }: { data: ResearchDetail; issues: string[]; cite: string | null; onCite: (ref: string | null) => void; sources: Sources | null }) {
  const b = data.budget
  const todo = (data.document?.match(/%\s*TODO/g) || []).length
  const templates = data.sources.filter((s) => s.id.startsWith('T'))
  return (
    <div className="ledger">
      {cite && <Passage refText={cite} data={data} sources={sources} onClose={() => onCite(null)} />}
      <section className="aside-section">
        <h3 className="aside-title">Checks</h3>
        <p className="small">{todo ? `${todo} TODO comment${todo === 1 ? '' : 's'} left for you to settle.` : data.document ? 'No TODO comments.' : 'Checks run once the document is assembled.'}</p>
        {issues.length > 0 && <ul className="issue-list">{issues.map((issue) => <li key={issue}>{issue}</li>)}</ul>}
      </section>
      <section className="aside-section">
        <h3 className="aside-title">Template</h3>
        {templates.length ? (
          <>
            <p className="small">{templates.map((t) => t.path.split('/').pop()).join(', ')}</p>
            {data.macros && data.macros.length > 0 && <p className="macro-list">{data.macros.map((m) => <code key={m}>\{m}</code>)}</p>}
            {data.theorems && data.theorems.length > 0 && <p className="muted small">Environments: {data.theorems.join(', ')}</p>}
          </>
        ) : <p className="muted small">No template: standard amsart with amsmath and amsthm.</p>}
      </section>
      <section className="aside-section">
        <h3 className="aside-title">Budget</h3>
        <Meter label="Generated tokens" used={b.tokens.used} limit={b.tokens.limit} />
        <Meter label="Input tokens" used={b.input_tokens.used} limit={b.input_tokens.limit} />
        <Meter label="Time, minutes" used={b.seconds.used / 60} limit={b.seconds.limit / 60} decimals={1} />
      </section>
    </div>
  )
}
