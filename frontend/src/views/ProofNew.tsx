import { useEffect, useState, type FormEvent } from 'react'
import { api } from '../api'
import { Collapse, ErrorNote, FilePicker, NumberField, Toggle } from '../components/common'
import { EffortSlider, effortLimits, type EffortLevel } from '../components/Effort'
import { Markdown } from '../components/Markdown'
import { useDropTarget } from '../drop'
import { go } from '../router'
import { useApp } from '../store'
import { BusyNote } from './shared'

export function ProofNew() {
  const { status, snapshot } = useApp()
  const defaults = status?.defaults
  const [goal, setGoal] = useState('')
  const [files, setFiles] = useState<string[]>([])
  const [rounds, setRounds] = useState(10)
  const [tokens, setTokens] = useState(60000)
  const [minutes, setMinutes] = useState(30)
  const [think, setThink] = useState(true)
  const [maxPredict, setMaxPredict] = useState(8192)
  const [ctx, setCtx] = useState(8192)
  const [literature, setLiterature] = useState(false)
  const [online, setOnline] = useState(false)
  const [level, setLevel] = useState<EffortLevel>('medium')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useDropTarget('Pinned to this proof as sources', (paths) => {
    const text = paths.filter((path) => !path.toLowerCase().endsWith('.pdf'))
    setFiles((items) => [...new Set([...items, ...text])])
    if (text.length < paths.length) setError('Proofs pin text sources only; the PDF was saved in the folder but not pinned.')
  })

  useEffect(() => {
    if (!defaults) return
    setRounds(defaults.proof_rounds)
    setTokens(defaults.proof_tokens)
    setMinutes(Math.round(defaults.proof_seconds / 60))
    setOnline(!!status?.online)
    setThink(defaults.think)
    setMaxPredict(defaults.proof_max_predict)
    setCtx(defaults.ctx)
    setLiterature(!!status?.proof_literature)
  }, [defaults, status?.proof_literature])

  const preset = defaults ? effortLimits('proof', defaults, level) : null
  const custom = !!preset && (rounds !== preset.rounds || tokens !== preset.tokens || minutes !== Math.round(preset.seconds / 60))
  const chooseLevel = (next: EffortLevel) => {
    setLevel(next)
    if (!defaults) return
    const limits = effortLimits('proof', defaults, next)
    setRounds(limits.rounds!)
    setTokens(limits.tokens)
    setMinutes(Math.round(limits.seconds / 60))
  }

  const running = snapshot.task && ['starting', 'running', 'pausing'].includes(snapshot.task.state)
  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      const result = await api.startProof({
        goal, source_files: files, rounds, tokens, seconds: minutes * 60, think,
        max_predict: maxPredict, ctx: ctx !== defaults?.ctx ? ctx : undefined, literature,
        online: literature && !status?.online_locked ? online : undefined,
      })
      if (result.id) go('prove', result.id)
      else setError(result.task.error || 'The proof did not start.')
    } catch (reason) {
      setError((reason as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="sheet paper">
      <form className="sheet-inner compose" onSubmit={submit} noValidate>
        <div className="entry">
          <span className="in-margin margin-glyph" aria-hidden="true">⊢</span>
          <div>
            <h1 className="page-title">Prove a statement</h1>
            <p className="lede">
              The harness keeps your statement fixed and works in bounded rounds: a solver writes an attempt,
              a fresh critic looks for the first unsupported step, and a recorder saves claims and objections.
              Every role is the same local model, so a finished proof is a candidate to check yourself, not a certificate.
            </p>
          </div>
        </div>

        <div className="entry">
          <span className="in-margin" />
          <label className="field">
            <span className="field-label">Statement and instructions</span>
            <textarea className="goal-input" rows={6} value={goal} onChange={(event) => setGoal(event.target.value)}
              placeholder={'Let $(u_n)$ be bounded in $H^1(0,1)$. Prove that $(u_n)$ has a subsequence converging strongly in $L^2(0,1)$.'}
              required />
            <span className="field-hint">Write LaTeX between dollar signs. The job does not see earlier conversations, so include every hypothesis.</span>
          </label>
        </div>

        {goal.includes('$') || goal.includes('\\(') ? (
          <div className="entry">
            <span className="in-margin muted small">Preview</span>
            <Markdown className="goal-preview">{goal}</Markdown>
          </div>
        ) : null}

        <div className="entry">
          <span className="in-margin" />
          <div className="field">
            <span className="field-label">Pinned sources</span>
            <FilePicker selected={files} onChange={setFiles} />
            <span className="field-hint">A snapshot of each pinned file is saved with the job and stays fixed even if you edit the file. File names written in the statement are pinned automatically.</span>
          </div>
        </div>

        <div className="entry">
          <span className="in-margin" />
          <div>
            {defaults && <EffortSlider kind="proof" defaults={defaults} level={level} onLevel={chooseLevel} custom={custom} />}
            <p className="field-hint">The limits are saved with the job. Resuming later continues from what is left; it never grants a new budget.</p>
            <Toggle label="Model thinking" checked={think} onChange={setThink}
              hint="Thinking tokens count toward the budget. The harness may still switch to direct written attempts after repeated truncation." />
            {status?.proof_literature ? (
              <>
                <Toggle label="Literature tools" checked={literature} onChange={setLiterature}
                  hint="The solver may read papers. This changes what the proof search measures." />
                {literature && (
                  <Toggle label="Search online" checked={online && !status.online_locked} onChange={setOnline} disabled={status.online_locked}
                    hint={status.online_locked ? 'Launched with --offline: cached papers only.' : online ? 'Search queries and downloads leave this computer.' : 'Cached papers only.'} />
                )}
              </>
            ) : (
              <p className="field-hint">Literature tools are off for proofs. Relaunch with --proof-literature to allow them.</p>
            )}
            <Collapse summary="Advanced limits">
              <div className="field-grid">
                <NumberField label="Work rounds" value={rounds} onChange={setRounds} min={1} max={100} />
                <NumberField label="Generated tokens" value={tokens} onChange={setTokens} step={1000} />
                <NumberField label="Time" value={minutes} onChange={setMinutes} min={1} suffix="min" />
                <NumberField label="Solver output ceiling" value={maxPredict} onChange={setMaxPredict} step={1024}
                  hint="Tokens per solver call after truncation" />
                <NumberField label="Context" value={ctx} onChange={setCtx} min={2048} step={2048}
                  hint="Larger context needs more memory" />
              </div>
            </Collapse>
          </div>
        </div>

        <div className="entry">
          <span className="in-margin" />
          <div className="form-actions">
            <ErrorNote>{error}</ErrorNote>
            {running && <BusyNote />}
            <button type="submit" className="btn btn-primary btn-large" disabled={busy || !goal.trim() || !!running}>
              {busy ? 'Starting…' : 'Start proof'}
            </button>
          </div>
        </div>
      </form>
    </div>
  )
}
