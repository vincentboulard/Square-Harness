import { useEffect, useState, type FormEvent } from 'react'
import { api, type Template } from '../api'
import { Collapse, ErrorNote, FilePicker, NumberField } from '../components/common'
import { EffortSlider, effortLimits, type EffortLevel } from '../components/Effort'
import { useDropTarget } from '../drop'
import { go } from '../router'
import { useApp } from '../store'
import { BusyNote, queuedMessage } from './shared'

const TEMPLATE = ['.sty', '.cls', '.tex', '.bib']

export function WriteupNew() {
  const { status, snapshot } = useApp()
  const d = status?.defaults
  const [goal, setGoal] = useState('')
  const [sources, setSources] = useState<string[]>([])
  const [notes, setNotes] = useState('')
  const [templateFiles, setTemplateFiles] = useState<string[]>([])
  const [templates, setTemplates] = useState<Template[]>([])
  const [saved, setSaved] = useState('')
  const [remember, setRemember] = useState('')
  const [output, setOutput] = useState('')
  const [tokens, setTokens] = useState(60000)
  const [minutes, setMinutes] = useState(15)
  const [inputTokens, setInputTokens] = useState(240000)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [queuedNote, setQueuedNote] = useState('')
  const [level, setLevel] = useState<EffortLevel>('medium')

  const applyLevel = (next: EffortLevel) => {
    if (!d) return
    const limits = effortLimits('writeup', d, next)
    setTokens(limits.tokens)
    setMinutes(Math.max(1, Math.round(limits.seconds / 60)))
    setInputTokens(limits.input_tokens!)
  }
  useEffect(() => {
    if (!d) return
    applyLevel(level)
    // Only when the launch defaults arrive.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [d])
  const preset = d ? effortLimits('writeup', d, level) : null
  const custom = !!preset && (tokens !== preset.tokens || minutes !== Math.max(1, Math.round(preset.seconds / 60)) || inputTokens !== preset.input_tokens)
  const chooseLevel = (next: EffortLevel) => {
    setLevel(next)
    applyLevel(next)
  }
  useEffect(() => { api.templates().then((result) => setTemplates(result.templates), () => undefined) }, [])
  useDropTarget('Notes go to the sources; .sty and .cls files go to the template', (paths) => {
    const style = paths.filter((path) => /\.(sty|cls)$/i.test(path))
    setTemplateFiles((items) => [...new Set([...items, ...style])])
    setSources((items) => [...new Set([...items, ...paths.filter((path) => !style.includes(path))])])
  })

  const running = !!snapshot.task && ['starting', 'running', 'pausing'].includes(snapshot.task.state)
  const hasMaterial = sources.length > 0 || notes.trim().length > 0
  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError('')
    setQueuedNote('')
    try {
      const result = await api.startWriteup({
        goal: goal.trim() || 'Write up the pinned notes as a clean LaTeX document.',
        source_files: sources, notes, template_files: templateFiles, template: saved || undefined,
        save_template: remember.trim() || undefined, output: output.trim() || undefined,
        tokens, seconds: minutes * 60, input_tokens: inputTokens, queue: true,
      })
      if (result.task.state === 'queued') {
        setQueuedNote(queuedMessage(result.task.title))
        setGoal('')
        setSources([])
        setNotes('')
      } else if (result.id) go('writeup', result.id)
      else setError(result.task.error || 'The write-up did not start.')
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
          <span className="in-margin margin-glyph" aria-hidden="true">§</span>
          <div>
            <h1 className="page-title">Write up your notes</h1>
            <p className="lede">
              Give rough notes, a draft or a PDF, and optionally your own macros and style. The harness plans sections,
              writes each from the exact lines of your notes, and marks every paragraph with the lines it came from.
              It adds no mathematics: unclear points become TODO comments for you to settle.
            </p>
            <p className={'network-note' + (status?.latex ? '' : ' network-online')}>
              {status?.latex
                ? 'LaTeX is installed here: the document is compiled, and sections with errors are repaired once.'
                : 'LaTeX (latexmk) was not found on this computer: you will get the .tex file without a compile check.'}
            </p>
          </div>
        </div>

        <div className="entry">
          <span className="in-margin" />
          <div className="field">
            <span className="field-label">Notes, drafts or PDFs</span>
            <FilePicker selected={sources} onChange={setSources} allowPdf accept={['.tex', '.md', '.txt', '.pdf']} label="Pin notes from the folder" />
            <span className="field-hint">Or drop files anywhere on this page. PDF text is extracted locally.</span>
          </div>
        </div>

        <div className="entry">
          <span className="in-margin" />
          <label className="field">
            <span className="field-label">Rough notes typed here</span>
            <textarea className="goal-input notes-input" rows={6} value={notes} onChange={(event) => setNotes(event.target.value)}
              placeholder={'u_n bounded in H^1(0,1) => subsequence u_n -> u weakly in H^1\nRellich: H^1 -> L^2 compact, so strongly in L^2\ncheck: need bounded interval!'} />
            <span className="field-hint">Saved with the job as its own source, so paragraphs can cite its lines.</span>
          </label>
        </div>

        <div className="entry">
          <span className="in-margin" />
          <div className="field template-slot">
            <span className="field-label">Your template</span>
            <p className="field-hint">Style files (.sty, .cls) with your macros, and optionally a template .tex whose preamble the document should use. The model must use your commands and theorem environments.</p>
            <FilePicker selected={templateFiles} onChange={setTemplateFiles} accept={TEMPLATE} label="Choose template files" />
            {templates.length > 0 && (
              <label className="field">
                <span className="field-label">Or a template you saved</span>
                <select value={saved} onChange={(event) => setSaved(event.target.value)}>
                  <option value="">None</option>
                  {templates.map((item) => <option key={item.name} value={item.name}>{item.name} ({item.files.join(', ')})</option>)}
                </select>
              </label>
            )}
            {templateFiles.length > 0 && (
              <label className="field">
                <span className="field-label">Remember these files as a template named</span>
                <input type="text" value={remember} onChange={(event) => setRemember(event.target.value)} placeholder="My article style" />
                <span className="field-hint">Saved for every folder you open in this interface.</span>
              </label>
            )}
          </div>
        </div>

        <div className="entry">
          <span className="in-margin" />
          <label className="field">
            <span className="field-label">Instructions</span>
            <textarea className="goal-input" rows={3} value={goal} onChange={(event) => setGoal(event.target.value)}
              placeholder="A short article with an introduction, the main lemma and its proof. Keep my notation." />
          </label>
        </div>

        <div className="entry">
          <span className="in-margin" />
          <div>
            <label className="field output-field">
              <span className="field-label">Output file</span>
              <span className="field-input"><input type="text" value={output} onChange={(event) => setOutput(event.target.value)} placeholder="notes-writeup.tex" /></span>
              <span className="field-hint">Created in this folder; an existing file is never replaced.</span>
            </label>
            {d && <EffortSlider kind="writeup" defaults={d} level={level} onLevel={chooseLevel} custom={custom} />}
            <Collapse summary="Advanced limits">
              <div className="field-grid">
                <NumberField label="Generated tokens" value={tokens} onChange={setTokens} step={1000} />
                <NumberField label="Time" value={minutes} onChange={setMinutes} min={1} suffix="min" />
                <NumberField label="Input tokens" value={inputTokens} onChange={setInputTokens} step={10000} hint="Every call's context counts again" />
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
            <button type="submit" className="btn btn-primary btn-large" disabled={busy || !hasMaterial}>
              {busy ? 'Starting…' : running ? 'Add write-up to the queue' : 'Start write-up'}
            </button>
            {!hasMaterial && <p className="field-hint">Pin or drop notes, or type them above.</p>}
          </div>
        </div>
      </form>
    </div>
  )
}
