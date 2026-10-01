import { useState } from 'react'
import { api, type ExperimentResults, type ExperimentRun, type ResearchDetail } from '../api'
import { Collapse, ErrorNote, Loading, Meter, Tabs, useLoad } from '../components/common'
import { PauseIcon, PlayIcon } from '../components/Icons'
import { Markdown } from '../components/Markdown'
import { researchLook, researchResumable, Square, type Variant } from '../components/Square'
import { ago } from '../format'
import { go } from '../router'
import { useApp, useTick } from '../store'
import { ISOLATION_TEXT } from './ExperimentNew'
import { Files } from './ProofView'
import { ActivityLog, ArtifactViewer, BusyNote, JobAside, ROLES, StreamBody, useJobClass, useLiveStream } from './shared'

const STEPS = ['plan', 'code', 'run', 'interpret', 'certificate', 'done'] as const
const STEP_NAMES: Record<string, string> = {
  plan: 'Protocol', code: 'Code', run: 'Run', interpret: 'Interpret', certificate: 'Certify', done: 'Done',
}
// Phases that belong to a step of the strip above.
const STEP_OF: Record<string, string> = { fix: 'run', check: 'certificate', faithful: 'certificate' }

const MEANING: Record<string, string> = {
  certified_counterexample: 'An explicit witness passed the exact checker, and a fresh review judged that it encodes the original statement. Read the certificate: that correspondence is a model judgement.',
  evidence_against: 'A validated, converged computation contradicts the claim. This is numerical evidence, not a rigorous refutation.',
  consistent: 'A validated, converged computation agrees with the claim on the explored family. This is not a proof, and says nothing outside that family.',
  inconclusive: 'The computation does not settle the question (unconverged, or no clear verdict).',
  validation_failed: 'The method failed on a case with a known answer: the other numbers are not reliable.',
  unvalidated: 'The code never checked its method on a case with a known answer: the numbers are not reliable.',
  run_failed: 'The experiment code did not run successfully within its runs.',
  not_run: 'The protocol and code were written but not run (experiments off, or the run was declined). You can read and run the code yourself.',
}

const num = (value: unknown, digits = 4) =>
  typeof value === 'number' ? (Math.abs(value) >= 1e5 || (Math.abs(value) < 1e-3 && value !== 0) ? value.toExponential(digits - 1) : Number(value.toPrecision(digits)).toString()) : String(value ?? '—')

export function ExperimentView({ id, tab }: { id: string; tab: string | null }) {
  const tick = useTick(id)
  const app = useApp()
  const { data, error } = useLoad(() => api.research(id), [id, tick])
  const [artifact, setArtifact] = useState<string | null>(null)
  const active = tab || 'result'
  const jobClass = useJobClass()
  const task = app.snapshot.task
  const ours = !!task && task.kind === 'research' && task.target === id && ['starting', 'running', 'pausing'].includes(task.state)
  if (!data) {
    return <div className="sheet paper"><div className="sheet-inner">{error ? <ErrorNote>{error}</ErrorNote> : <Loading />}</div></div>
  }
  const runs = data.runs || []
  const run = lastRun(runs)
  const results = run?.result?.results || null
  return (
    <div className={jobClass}>
      <div className="job-main sheet paper">
        <div className="sheet-inner">
          <Header data={data} ours={ours} pausing={task?.state === 'pausing'} />
          <Tabs active={active} onChange={(next) => go('experiment', id, next === 'result' ? null : next)} tabs={[
            { id: 'result', label: 'Result' },
            { id: 'checks', label: 'Checks', narrowOnly: true },
            { id: 'figures', label: 'Figures', badge: figures(run).length || undefined },
            { id: 'code', label: 'Code' },
            { id: 'runs', label: 'Runs', badge: runs.length || undefined },
            { id: 'protocol', label: 'Protocol' },
            { id: 'files', label: 'Files', badge: data.artifacts.length || undefined },
          ]} />
          {active === 'result' && <Result data={data} run={run} results={results} />}
          {active === 'checks' && <div className="ledger-tab"><Checks data={data} results={results} /></div>}
          {active === 'figures' && <Figures id={id} run={run} />}
          {active === 'code' && <Code data={data} />}
          {active === 'runs' && <Runs runs={runs} />}
          {active === 'protocol' && <Protocol data={data} />}
          {active === 'files' && <Files names={data.artifacts} onOpen={setArtifact} />}
        </div>
      </div>
      <JobAside label="Checks"><Checks data={data} results={results} /></JobAside>
      {artifact && <ArtifactViewer job="research" id={id} name={artifact} onClose={() => setArtifact(null)} />}
    </div>
  )
}

function lastRun(runs: ExperimentRun[]) {
  const done = runs.filter((r) => r.result)
  return [...done].reverse().find((r) => r.result!.ok) || done[done.length - 1] || null
}

const figures = (run: ExperimentRun | null) => (run?.result?.figures || []).filter((name) => name.endsWith('.png'))

function Header({ data, ours, pausing }: { data: ResearchDetail; ours: boolean; pausing: boolean }) {
  const look = researchLook(data.status, data.running)
  const app = useApp()
  const [error, setError] = useState('')
  const busy = !!app.snapshot.task && ['starting', 'running', 'pausing'].includes(app.snapshot.task.state)
  const queued = app.snapshot.queue.some((item) => item.target === data.id)
  const act = (action: Promise<unknown>) => { setError(''); action.catch((reason: Error) => setError(reason.message)) }
  const phase = STEP_OF[data.phase] || data.phase
  const current = STEPS.indexOf(phase as typeof STEPS[number])
  return (
    <header className="job-head">
      <div className="job-status">
        <Square variant={look.variant} size={18} label={look.label} />
        <span className="job-status-label">{look.label}</span>
        <span className="job-stop">Experiment</span>
      </div>
      <div className="job-goal"><Markdown>{data.goal}</Markdown></div>
      <ol className="phases phases-wide" aria-label="Workflow">
        {STEPS.map((step, index) => {
          const variant: Variant = index < current || data.phase === 'done' ? 'complete' : index === current ? (data.running ? 'running' : 'paused') : 'ready'
          return <li key={step} className={'phase phase-' + variant}><Square variant={variant} size={10} />{STEP_NAMES[step]}</li>
        })}
      </ol>
      <div className="job-facts">
        {data.sources.map((source) => <span key={source.id} className="fact-file">{source.id} {source.path}</span>)}
        <span>{String(data.settings.model)}</span>
        <span>Code: {data.experiment?.permission === 'off' ? 'not run' : data.experiment?.permission === 'auto' ? 'runs automatically' : 'asks before each run'}</span>
        <span>Started {ago(data.created_at)}</span>
      </div>
      <div className="job-actions">
        {ours ? (
          <button type="button" className="btn" disabled={pausing} onClick={() => act(api.pause())}><PauseIcon size={16} /> {pausing ? 'Pausing…' : 'Pause'}</button>
        ) : researchResumable(data.status) ? (
          <button type="button" className="btn btn-primary" disabled={queued} onClick={() => act(api.resumeResearch(data.id, busy))}>
            <PlayIcon size={16} /> {queued ? 'Waiting in the queue' : busy ? 'Resume when the model is free' : 'Resume with the remaining budget'}
          </button>
        ) : null}
        {data.running && !ours && <span className="muted">Running in another process, such as a terminal.</span>}
      </div>
      {!ours && busy && researchResumable(data.status) && !queued && <BusyNote queue />}
      <ErrorNote>{error}</ErrorNote>
    </header>
  )
}

function Live({ data }: { data: ResearchDetail }) {
  const stream = useLiveStream('research', data.id, data.live?.file || null)
  const app = useApp()
  const task = app.snapshot.task
  const ours = !!task && task.kind === 'research' && task.target === data.id
  return (
    <div className="live" id="live">
      <p className="live-head"><Square variant="running" size={14} /> {data.live ? (ROLES[data.live.role]?.active || data.live.role) : 'Running the experiment code'}</p>
      {ours && <ActivityLog items={app.snapshot.activity} />}
      {data.live && <StreamBody stream={stream} role={data.live.role} live />}
    </div>
  )
}

function Result({ data, run, results }: { data: ResearchDetail; run: ExperimentRun | null; results: ExperimentResults | null }) {
  const interpretation = data.interpretation
  const finished = !data.running && MEANING[data.status]
  return (
    <div className="report">
      {data.running && <Live data={data} />}
      {finished && (
        <div className={'verdict verdict-' + data.status}>
          <p className="verdict-title">{researchLook(data.status).label}</p>
          <p>{MEANING[data.status]}</p>
        </div>
      )}
      {data.warnings.length > 0 && (
        <div className="checks-note">
          <p className="objection-label">Controller notes</p>
          <ul>{data.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
        </div>
      )}
      {data.certificate && <Certificate data={data} />}
      {interpretation && (
        <section className="experiment-section">
          <h2 className="section-title">Interpretation <span className="muted small">model, read against the protocol</span></h2>
          <p className="small">Verdict: <strong>{interpretation.verdict}</strong></p>
          <Markdown>{interpretation.explanation}</Markdown>
          {interpretation.key_numbers.length > 0 && <ul className="key-numbers">{interpretation.key_numbers.map((n) => <li key={n}><Markdown>{n}</Markdown></li>)}</ul>}
          {interpretation.limitations && <p className="caveat"><strong>Limitations.</strong> {interpretation.limitations}</p>}
        </section>
      )}
      {results && <Tables results={results} />}
      {run && figures(run).length > 0 && (
        <section className="experiment-section">
          <h2 className="section-title">Figures</h2>
          <FigureGrid id={data.id} run={run} />
        </section>
      )}
      {!interpretation && !data.running && !data.certificate && (
        <p className="empty-note">{data.code ? 'No interpretation was recorded. The code and any output are under Code and Runs.' : 'Nothing was recorded yet.'}</p>
      )}
    </div>
  )
}

function Certificate({ data }: { data: ResearchDetail }) {
  const check = data.certificate_check
  return (
    <section className={'experiment-section certificate' + (check?.certified ? ' certificate-ok' : '')}>
      <h2 className="section-title">Counterexample certificate</h2>
      <p className="small"><strong>{check?.certified ? 'Certified by the exact checker' : 'Not certified'}</strong>{check ? ' — ' + check.reason : ''}</p>
      {check?.checks && (
        <ul className="check-list">
          {check.checks.map((item) => (
            <li key={item.what} className={item.holds === (item.what === 'claim at the witness') ? 'check-bad' : 'check-good'}>
              <span className="check-what">{item.what}</span> <code>{item.relation}</code> → {item.holds ? 'holds' : 'fails'}
              <span className="muted small"> ({item.method}; {item.detail})</span>
            </li>
          ))}
        </ul>
      )}
      {data.faithfulness && (
        <p className="small">Encodes the statement (model review): <strong>{data.faithfulness.verdict.replace('_', ' ')}</strong>. {data.faithfulness.explanation}</p>
      )}
      <Collapse summary="Certificate (JSON)"><pre className="source">{JSON.stringify(data.certificate, null, 2)}</pre></Collapse>
      <p className="field-hint">Rerun the check yourself: <code>python verify_certificate.py certificate.json</code> in the job’s <code>certificate</code> folder.</p>
    </section>
  )
}

function Tables({ results }: { results: ExperimentResults }) {
  return (
    <>
      {!!results.validations?.length && (
        <section className="experiment-section">
          <h2 className="section-title">Validation on known cases</h2>
          <table className="result-table">
            <thead><tr><th>Case</th><th>Max relative error</th><th>Tolerance</th><th>Passed</th></tr></thead>
            <tbody>{results.validations.map((v) => (
              <tr key={v.name} className={v.passed ? '' : 'row-bad'}><td>{v.name}</td><td>{num(v.max_relative_error, 3)}</td><td>{num(v.rtol, 2)}</td><td>{v.passed ? 'yes' : 'no'}</td></tr>
            ))}</tbody>
          </table>
        </section>
      )}
      {!!results.convergence?.length && (
        <section className="experiment-section">
          <h2 className="section-title">Convergence</h2>
          <table className="result-table">
            <thead><tr><th>Quantity</th><th>Observed order</th><th>Last relative change</th><th>Extrapolated</th><th>Converged</th></tr></thead>
            <tbody>{results.convergence.map((c) => (
              <tr key={c.name} className={c.converged ? '' : 'row-bad'}><td>{c.name}</td><td>{num(c.observed_order, 3)}</td><td>{num(c.last_relative_change, 3)}</td><td>{num(c.extrapolated, 10)}</td><td>{c.converged ? 'yes' : 'no'}</td></tr>
            ))}</tbody>
          </table>
        </section>
      )}
      {!!results.searches?.length && (
        <section className="experiment-section">
          <h2 className="section-title">Adversarial searches</h2>
          <ul className="key-numbers">{results.searches.map((s) => (
            <li key={s.name}>{s.name}: max {num(s.max, 10)} at [{(s.argmax || []).map((x) => num(x, 6)).join(', ')}], {s.evaluations} evaluations</li>
          ))}</ul>
        </section>
      )}
      {results.values && Object.keys(results.values).length > 0 && (
        <Collapse summary={`Recorded values (${Object.keys(results.values).length})`}>
          <pre className="plain">{JSON.stringify(results.values, null, 1)}</pre>
        </Collapse>
      )}
    </>
  )
}

function FigureGrid({ id, run }: { id: string; run: ExperimentRun }) {
  return (
    <div className="figure-grid">
      {figures(run).map((name) => {
        const svg = name.replace(/\.png$/, '.svg')
        const hasSvg = run.result?.figures.includes(svg)
        return (
          <figure key={name} className="figure">
            <img src={api.figureUrl(id, run.folder!, name)} alt={name} loading="lazy" />
            <figcaption>
              {name.replace(/\.png$/, '')}
              {hasSvg && <> · <a href={api.figureUrl(id, run.folder!, svg)}>SVG</a></>}
            </figcaption>
          </figure>
        )
      })}
    </div>
  )
}

function Figures({ id, run }: { id: string; run: ExperimentRun | null }) {
  if (!run || !figures(run).length) return <p className="empty-note">No figures were saved.</p>
  return <FigureGrid id={id} run={run} />
}

function Code({ data }: { data: ResearchDetail }) {
  if (!data.code) return <p className="empty-note">No code yet.</p>
  const env = data.experiment?.environment
  return (
    <div>
      <p className="caveat">The final script. It imports the harness’s tested helpers as <code>sq</code>; each run folder keeps a copy you can rerun with <code>python experiment.py</code>.</p>
      {data.code_issue && <div className="checks-note"><p className="objection-label">{data.code_issue}</p></div>}
      <pre className="source">{data.code}</pre>
      {env && <p className="field-hint">Python {env.python}; {Object.entries(env.packages).map(([k, v]) => `${k} ${v || 'missing'}`).join(', ')}.</p>}
    </div>
  )
}

function Runs({ runs }: { runs: ExperimentRun[] }) {
  if (!runs.length) return <p className="empty-note">No run yet.</p>
  return (
    <div className="runs">
      {[...runs].reverse().map((run) => {
        const r = run.result
        return (
          <section key={run.index} className="experiment-section">
            <h2 className="section-title">Run {run.index} <span className="muted small">{r ? `exit ${r.returncode}${r.limit_reason ? ', ' + r.limit_reason : ''}, ${r.seconds} s, isolation ${r.isolation}` : 'not run'}</span></h2>
            {run.permission && <p className="small muted">{run.permission}</p>}
            {run.skipped && <p className="small">{run.skipped}</p>}
            {r && <>
              <Collapse summary="Output" open={!r.ok}><pre className="plain">{r.stdout || '(no output)'}</pre></Collapse>
              {r.stderr.trim() && <Collapse summary="Errors and warnings" open={!r.ok}><pre className="plain">{r.stderr}</pre></Collapse>}
            </>}
          </section>
        )
      })}
    </div>
  )
}

function Protocol({ data }: { data: ResearchDetail }) {
  if (!data.protocol) return <p className="empty-note">No protocol yet.</p>
  const labels: Record<string, string> = {
    hypothesis: 'Hypothesis', quantity: 'Quantity computed', support_criterion: 'Would support the claim',
    refutation_criterion: 'Would refute the claim', validation_case: 'Validation case (known answer)',
    discretisation_control: 'Discretisation control', search_family: 'Explored family', figures: 'Figures', raw: 'Protocol (unstructured)',
  }
  return (
    <div>
      <p className="caveat">Fixed before any computation; the interpretation is read against it.</p>
      <dl className="protocol">
        {Object.entries(data.protocol).map(([key, value]) => (
          <div key={key}><dt>{labels[key] || key}</dt><dd><Markdown>{String(value)}</Markdown></dd></div>
        ))}
      </dl>
      {data.experiment?.context && <Collapse summary="Context given to the experiment"><Markdown>{data.experiment.context}</Markdown></Collapse>}
    </div>
  )
}

function Checks({ data, results }: { data: ResearchDetail; results: ExperimentResults | null }) {
  const b = data.budget
  const validations = results?.validations || []
  const convergence = results?.convergence || []
  const isolation = data.experiment?.environment?.isolation || 'none'
  return (
    <div className="ledger">
      <section className="aside-section">
        <h3 className="aside-title">Reliability</h3>
        <ul className="check-list">
          <li className={validations.length && validations.every((v) => v.passed) ? 'check-good' : 'check-bad'}>
            {validations.length ? `${validations.filter((v) => v.passed).length}/${validations.length} validations passed` : 'No validation recorded'}
          </li>
          <li className={convergence.every((c) => c.converged) ? 'check-good' : 'check-bad'}>
            {convergence.length ? `${convergence.filter((c) => c.converged).length}/${convergence.length} quantities converged` : 'No convergence study recorded'}
          </li>
          {data.certificate_check && <li className={data.certificate_check.certified ? 'check-good' : 'check-bad'}>
            Certificate {data.certificate_check.certified ? 'certified' : 'not certified'}
          </li>}
        </ul>
      </section>
      <section className="aside-section">
        <h3 className="aside-title">Isolation</h3>
        <p className="small">{ISOLATION_TEXT[isolation]}</p>
      </section>
      <section className="aside-section">
        <h3 className="aside-title">Budget</h3>
        <Meter label="Code runs" used={(data.runs || []).filter((r) => r.result).length} limit={b.rounds.limit} />
        <Meter label="Generated tokens" used={b.tokens.used} limit={b.tokens.limit} />
        <Meter label="Input tokens" used={b.input_tokens.used} limit={b.input_tokens.limit} />
        <Meter label="Time, minutes" used={b.seconds.used / 60} limit={b.seconds.limit / 60} decimals={1} />
      </section>
    </div>
  )
}
