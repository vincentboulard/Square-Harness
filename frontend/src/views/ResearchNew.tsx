import { useEffect, useState, type FormEvent } from 'react'
import { api } from '../api'
import { Collapse, ErrorNote, FilePicker, NumberField, Toggle } from '../components/common'
import { EffortSlider, effortLimits, type EffortLevel } from '../components/Effort'
import { useDropTarget } from '../drop'
import { go } from '../router'
import { Glyph } from '../components/Glyph'
import { useApp } from '../store'
import { BusyNote, queuedMessage } from './shared'

type Kind = 'literature' | 'referee'

const COPY: Record<Kind, { title: string; lede: string; placeholder: string; pinHint: string }> = {
  literature: {
    title: 'Build a reading list',
    lede: 'The harness searches Crossref, arXiv, zbMATH and Semantic Scholar, follows the citations of the key works, and lets the model screen the candidates, group them into entry points and themes, and say why each one is there. Every entry is a real record with its identifier; nothing is recalled from memory.',
    placeholder: 'Observability and control of the heat equation: what should a graduate student read, from the classical results to recent work?',
    pinHint: 'Optional. A pinned manuscript helps scope the topic; it is read locally and never uploaded.',
  },
  referee: {
    title: 'Review a manuscript',
    lede: 'Pin the manuscript. The harness maps its main claims, checks the consequential steps with line-level citations, and drafts a review that separates demonstrated errors, missing justifications and unresolved concerns.',
    placeholder: 'Assess the main theorem and its proof. Identify the first unjustified step and check related literature where available.',
    pinHint: 'Pin the manuscript (TeX, Markdown, text or PDF). Its snapshot stays fixed for the whole job.',
  },
}

export function ResearchNew({ kind }: { kind: Kind }) {
  const { status, snapshot } = useApp()
  const copy = COPY[kind]
  const d = status?.defaults
  const [goal, setGoal] = useState('')
  const [files, setFiles] = useState<string[]>([])
  const [rounds, setRounds] = useState(3)
  const [tokens, setTokens] = useState(60000)
  const [minutes, setMinutes] = useState(15)
  const [inputTokens, setInputTokens] = useState(240000)
  const [requests, setRequests] = useState(12)
  const [chars, setChars] = useState(30000)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [queuedNote, setQueuedNote] = useState('')
  const [online, setOnline] = useState(true)
  const [level, setLevel] = useState<EffortLevel>('medium')

  useDropTarget(kind === 'referee' ? 'Pinned as the manuscript' : 'Pinned as sources for the report',
    (paths) => setFiles((items) => [...new Set([...items, ...paths])]))

  const applyLevel = (next: EffortLevel) => {
    if (!d) return
    const limits = effortLimits('research', d, next)
    setRounds(limits.rounds!)
    setTokens(limits.tokens)
    setMinutes(Math.max(1, Math.round(limits.seconds / 60)))
    setInputTokens(limits.input_tokens!)
    setRequests(limits.requests!)
    setChars(limits.chars!)
  }
  useEffect(() => {
    if (!d) return
    applyLevel(level)
    // Only when the launch defaults arrive.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [d])
  // Online search is on unless the interface was launched with --offline.
  useEffect(() => { setOnline(!!status?.online) }, [status?.online])

  const preset = d ? effortLimits('research', d, level) : null
  const custom = !!preset && (rounds !== preset.rounds || tokens !== preset.tokens || minutes !== Math.max(1, Math.round(preset.seconds / 60))
    || inputTokens !== preset.input_tokens || requests !== preset.requests || chars !== preset.chars)
  const chooseLevel = (next: EffortLevel) => {
    setLevel(next)
    applyLevel(next)
  }
  const searching = online && !status?.online_locked

  const running = !!snapshot.task && ['starting', 'running', 'pausing'].includes(snapshot.task.state)
  const noun = kind === 'referee' ? 'review' : 'literature report'
  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError('')
    setQueuedNote('')
    try {
      const result = await api.startResearch({
        kind, goal, source_files: files, rounds, tokens, seconds: minutes * 60,
        input_tokens: inputTokens, requests, chars, online: status?.online_locked ? undefined : online, queue: true,
      })
      if (result.task.state === 'queued') {
        setQueuedNote(queuedMessage(result.task.title))
        setGoal('')
        setFiles([])
      } else if (result.id) go(kind, result.id)
      else setError(result.task.error || `The ${noun} did not start.`)
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
          <span className="in-margin margin-glyph" aria-hidden="true"><Glyph mode={kind} /></span>
          <div>
            <h1 className="page-title">{copy.title}</h1>
            <p className="lede">{copy.lede}</p>
            <p className={'network-note' + (searching ? ' network-online' : '')}>
              {searching
                ? `Online search is on for this ${noun}: search queries and paper downloads leave this computer. Search uses short public topic queries; manuscript text stays local. Untick Search online below to keep it offline.`
                : `Online search is off: the ${noun} uses pinned files and papers already cached in this workspace, and will say that its coverage is limited.`}
            </p>
          </div>
        </div>
        <div className="entry">
          <span className="in-margin" />
          <label className="field">
            <span className="field-label">Scope and request</span>
            <textarea className="goal-input" rows={5} value={goal} onChange={(event) => setGoal(event.target.value)} placeholder={copy.placeholder} required />
          </label>
        </div>
        <div className="entry">
          <span className="in-margin" />
          <div className="field">
            <span className="field-label">{kind === 'referee' ? 'Manuscript' : 'Pinned sources'}</span>
            <FilePicker selected={files} onChange={setFiles} allowPdf />
            <span className="field-hint">{copy.pinHint}</span>
          </div>
        </div>
        <div className="entry">
          <span className="in-margin" />
          <div>
            {d && <EffortSlider kind="research" defaults={d} level={level} onLevel={chooseLevel} custom={custom} />}
            <Toggle label="Search online" checked={searching} onChange={setOnline} disabled={status?.online_locked}
              hint={status?.online_locked ? 'Launched with --offline: online search stays off.' : (kind === 'literature' ? 'Crossref, arXiv, zbMATH and Semantic Scholar.' : 'arXiv, Semantic Scholar and zbMATH; web search needs a Brave key.')} />
            <Collapse summary="Advanced limits">
              <div className="field-grid">
                <NumberField label="Investigation rounds" value={rounds} onChange={setRounds} min={1} max={100} />
                <NumberField label="Generated tokens" value={tokens} onChange={setTokens} step={1000} />
                <NumberField label="Time" value={minutes} onChange={setMinutes} min={1} suffix="min" />
                <NumberField label="Input tokens" value={inputTokens} onChange={setInputTokens} step={10000} hint="Every call's context counts again" />
                <NumberField label="Web requests" value={requests} onChange={setRequests} min={0} max={100} hint={searching ? 'External retrievals' : 'Unused while offline'} />
                <NumberField label="Evidence characters" value={chars} onChange={setChars} min={1000} max={1000000} step={1000} hint="Returned by literature tools" />
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
            <button type="submit" className="btn btn-primary btn-large" disabled={busy || !goal.trim() || (kind === 'referee' && !files.length && !/\.(tex|md|txt|pdf)\b/.test(goal))}>
              {busy ? 'Starting…' : running ? `Add ${noun} to the queue` : `Start ${noun}`}
            </button>
            {kind === 'referee' && !files.length && <p className="field-hint">Pin the manuscript, or name its file in the request.</p>}
          </div>
        </div>
      </form>
    </div>
  )
}
