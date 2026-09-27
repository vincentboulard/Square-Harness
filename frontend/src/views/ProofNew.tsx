import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { api, type WorkspaceFile } from '../api'
import { Collapse, ErrorNote, FilePicker, NumberField } from '../components/common'
import { EffortSlider, effortLimits, type EffortLevel } from '../components/Effort'
import { Markdown } from '../components/Markdown'
import { useDropTarget } from '../drop'
import { go } from '../router'
import { useApp, useTick } from '../store'
import { BusyNote } from './shared'

// File names in the statement, as the terminal notices them (proof jobs have no file tools).
const MENTIONED = /(?<![\w/])[\w./-]+\.(?:tex|md|txt)\b/g

export function ProofNew() {
  const { status, snapshot } = useApp()
  const defaults = status?.defaults
  const [goal, setGoal] = useState('')
  const [files, setFiles] = useState<string[]>([])
  const [rounds, setRounds] = useState(3)
  const [tokens, setTokens] = useState(120000)
  const [minutes, setMinutes] = useState(30)
  const [solveTokens, setSolveTokens] = useState(32768)
  const [verifyTokens, setVerifyTokens] = useState(16384)
  const [ctx, setCtx] = useState(40960)
  const [level, setLevel] = useState<EffortLevel>('medium')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [workspace, setWorkspace] = useState<WorkspaceFile[]>([])
  const changed = useTick('files')

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
    setSolveTokens(defaults.proof_solve_tokens)
    setVerifyTokens(defaults.proof_verify_tokens)
    setCtx(defaults.ctx)
  }, [defaults])

  useEffect(() => {
    api.files().then((result) => setWorkspace(result.files), () => setWorkspace([]))
  }, [changed])

  const unpinned = useMemo(() => {
    const known = new Set(workspace.filter((file) => file.kind === 'text').map((file) => file.path))
    return [...new Set(goal.match(MENTIONED) || [])].filter((name) => known.has(name) && !files.includes(name))
  }, [goal, files, workspace])

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
        goal, source_files: files, rounds, tokens, seconds: minutes * 60,
        solve_tokens: solveTokens, verify_tokens: verifyTokens, ctx: ctx !== defaults?.ctx ? ctx : undefined,
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
              The harness keeps your statement fixed. The solver writes a complete proof with thinking on, a fresh
              verifier reviews the whole written proof, and a concrete objection leads to a repair or a rebuttal that is
              reviewed again. Both roles are the same model in separate contexts, so an approved proof is a candidate
              to check yourself, not a certificate.
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
            <span className="field-hint">The model sees only the statement and the pinned files: proof work has no file or web tools. A snapshot of each pinned file is saved with the job and stays fixed even if you edit the file.</span>
            {unpinned.length > 0 && (
              <p className="warning-note">
                Named in the statement but not pinned, so the model will not see {unpinned.length === 1 ? 'it' : 'them'}: {unpinned.join(', ')}.{' '}
                <button type="button" className="link-btn" onClick={() => setFiles([...files, ...unpinned])}>Pin {unpinned.length === 1 ? 'it' : 'them'}</button>
              </p>
            )}
          </div>
        </div>

        <div className="entry">
          <span className="in-margin" />
          <div>
            {defaults && <EffortSlider kind="proof" defaults={defaults} level={level} onLevel={chooseLevel} custom={custom} />}
            <p className="field-hint">The limits are saved with the job. Resuming later continues from what is left; it never grants a new budget. Thinking is always on for proofs and counts toward the output ceilings.</p>
            <Collapse summary="Advanced limits">
              <div className="field-grid">
                <NumberField label="Attempts" value={rounds} onChange={setRounds} min={1} max={100}
                  hint="The initial solve and each repair or continuation" />
                <NumberField label="Generated tokens" value={tokens} onChange={setTokens} step={1000}
                  hint={`At least one solve and one review: ${(solveTokens + verifyTokens).toLocaleString()}`} />
                <NumberField label="Time" value={minutes} onChange={setMinutes} min={1} suffix="min" />
                <NumberField label="Solve output ceiling" value={solveTokens} onChange={setSolveTokens} min={128} step={1024}
                  hint="Tokens per solve, including thinking" />
                <NumberField label="Review output ceiling" value={verifyTokens} onChange={setVerifyTokens} min={128} step={1024}
                  hint="Tokens per whole-proof review, including thinking" />
                <NumberField label="Context" value={ctx} onChange={setCtx} min={2048} step={2048}
                  hint="Must hold the problem, the pinned text and a full answer" />
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
