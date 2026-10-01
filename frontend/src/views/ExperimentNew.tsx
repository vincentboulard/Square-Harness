import { useEffect, useState, type FormEvent } from 'react'
import { api, type Permission } from '../api'
import { Collapse, ErrorNote, FilePicker, NumberField } from '../components/common'
import { EffortSlider, effortLimits, type EffortLevel } from '../components/Effort'
import { MODES } from '../modes'
import { useDropTarget } from '../drop'
import { go } from '../router'
import { useApp } from '../store'
import { BusyNote, queuedMessage } from './shared'

export const ISOLATION_TEXT: Record<string, string> = {
  bwrap: 'Runs are isolated with bubblewrap: no network, read-only system, home folder hidden, writes only in the run folder.',
  seatbelt: 'Runs are isolated with the macOS sandbox: no network, writes only in the run folder.',
  netns: 'Runs have no network access (Linux network namespace). The file system is not protected: review the code before approving it, or install bubblewrap for stronger isolation.',
  none: 'No isolation is available on this computer: code runs with your account’s permissions and resource limits only. Review every script before approving it.',
}

const PERMISSIONS: { id: Permission; label: string; hint: string }[] = [
  { id: 'ask', label: 'Ask before each run', hint: 'You see each script and approve or decline it.' },
  { id: 'auto', label: 'Run automatically', hint: 'Runs without asking when network isolation is available; otherwise asks.' },
  { id: 'off', label: 'Write code only', hint: 'The protocol and the script are saved; nothing runs.' },
]

export function PermissionChoice({ value, onChange }: { value: Permission; onChange: (value: Permission) => void }) {
  return (
    <div className="field">
      <span className="field-label">Running code</span>
      <div className="segmented" role="group" aria-label="Running code">
        {PERMISSIONS.map((item) => (
          <button key={item.id} type="button" aria-pressed={value === item.id} title={item.hint} onClick={() => onChange(item.id)}>
            {item.label}
          </button>
        ))}
      </div>
      <span className="field-hint">{PERMISSIONS.find((item) => item.id === value)!.hint}</span>
    </div>
  )
}

export function ExperimentNew() {
  const { status, snapshot } = useApp()
  const d = status?.defaults
  const [goal, setGoal] = useState('')
  const [files, setFiles] = useState<string[]>([])
  const [level, setLevel] = useState<EffortLevel>('medium')
  const [rounds, setRounds] = useState(4)
  const [tokens, setTokens] = useState(60000)
  const [inputTokens, setInputTokens] = useState(240000)
  const [minutes, setMinutes] = useState(15)
  const [runSeconds, setRunSeconds] = useState(120)
  const [memory, setMemory] = useState(2048)
  const [permission, setPermission] = useState<Permission>('ask')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [queuedNote, setQueuedNote] = useState('')

  useDropTarget('Pinned as sources for the experiment', (paths) => {
    const text = paths.filter((path) => !path.toLowerCase().endsWith('.pdf'))
    setFiles((items) => [...new Set([...items, ...text])])
  })

  const applyLevel = (next: EffortLevel) => {
    if (!d) return
    const limits = effortLimits('experiment', d, next)
    setRounds(limits.rounds!)
    setTokens(limits.tokens)
    setInputTokens(limits.input_tokens!)
    setMinutes(Math.max(1, Math.round(limits.seconds / 60)))
  }
  useEffect(() => {
    if (!d) return
    applyLevel(level)
    setRunSeconds(d.experiment_seconds)
    setMemory(d.experiment_memory)
    setPermission(d.experiments)
    // Only when the launch defaults arrive.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [d])

  const preset = d ? effortLimits('experiment', d, level) : null
  const custom = !!preset && (rounds !== preset.rounds || tokens !== preset.tokens || inputTokens !== preset.input_tokens
    || minutes !== Math.max(1, Math.round(preset.seconds / 60)))
  const chooseLevel = (next: EffortLevel) => { setLevel(next); applyLevel(next) }
  const running = !!snapshot.task && ['starting', 'running', 'pausing'].includes(snapshot.task.state)
  const isolation = status?.isolation || 'none'

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError('')
    setQueuedNote('')
    try {
      const result = await api.startExperiment({
        goal, source_files: files, permission, rounds, tokens, input_tokens: inputTokens, seconds: minutes * 60,
        run_seconds: runSeconds, memory_mb: memory, queue: true,
      })
      if (result.task.state === 'queued') {
        setQueuedNote(queuedMessage(result.task.title))
        setGoal('')
        setFiles([])
      } else if (result.id) go('experiment', result.id)
      else setError(result.task.error || 'The experiment did not start.')
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
          <span className="in-margin margin-glyph" aria-hidden="true">{MODES.experiment.glyph}</span>
          <div>
            <h1 className="page-title">Test a claim numerically</h1>
            <p className="lede">
              The harness fixes a protocol first (what would support or refute the claim, a validation case with a known
              answer, how discretisation error is controlled), then writes code with a tested numerical library, runs it,
              and reads the recorded numbers. The final status comes from the data, not from the model: unvalidated or
              unconverged results are reported as unreliable. Numerical evidence is never a proof; an explicit
              counterexample is checked in exact or interval arithmetic before it is called certified.
            </p>
            <p className={'network-note' + (isolation === 'none' || isolation === 'netns' ? ' network-online' : '')}>{ISOLATION_TEXT[isolation]}</p>
          </div>
        </div>
        <div className="entry">
          <span className="in-margin" />
          <label className="field">
            <span className="field-label">Claim, constant or intuition to test</span>
            <textarea className="goal-input" rows={5} value={goal} onChange={(event) => setGoal(event.target.value)} required
              placeholder="Is the best constant C in ‖u‖²_{L²(0,1)} ≤ C ‖u′‖²_{L²(0,1)} for u ∈ H¹₀(0,1) equal to 1/π²? Search for functions that come close." />
          </label>
        </div>
        <div className="entry">
          <span className="in-margin" />
          <div className="field">
            <span className="field-label">Pinned sources</span>
            <FilePicker selected={files} onChange={setFiles} />
            <span className="field-hint">Optional: notes or a statement the protocol should follow. Experiments never search online.</span>
          </div>
        </div>
        <div className="entry">
          <span className="in-margin" />
          <div>
            <PermissionChoice value={permission} onChange={setPermission} />
            {d && <EffortSlider kind="experiment" defaults={d} level={level} onLevel={chooseLevel} custom={custom} />}
            <Collapse summary="Advanced limits">
              <div className="field-grid">
                <NumberField label="Code runs" value={rounds} onChange={setRounds} min={1} max={20} hint="The first run and each fix" />
                <NumberField label="Seconds per run" value={runSeconds} onChange={setRunSeconds} min={5} max={3600} hint="CPU and wall-clock limit" />
                <NumberField label="Memory per run" value={memory} onChange={setMemory} min={256} max={65536} step={256} suffix="MB" />
                <NumberField label="Generated tokens" value={tokens} onChange={setTokens} min={4000} step={1000} />
                <NumberField label="Input tokens" value={inputTokens} onChange={setInputTokens} min={8000} step={10000} />
                <NumberField label="Time" value={minutes} onChange={setMinutes} min={1} suffix="min" />
              </div>
            </Collapse>
          </div>
        </div>
        <div className="entry">
          <span className="in-margin" />
          <div className="form-actions">
            <ErrorNote>{error}</ErrorNote>
            {queuedNote && <p className="queued-note" role="status">{queuedNote}</p>}
            {running && <BusyNote queue />}
            <button type="submit" className="btn btn-primary btn-large" disabled={busy || !goal.trim()}>
              {busy ? 'Starting…' : running ? 'Add the experiment to the queue' : 'Start the experiment'}
            </button>
          </div>
        </div>
      </form>
    </div>
  )
}
