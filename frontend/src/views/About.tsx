// The notebook's title page: what Square Harness is, who made it, and where to find more.
import { MODES } from '../modes'
import { href, RAIL, type Mode } from '../router'
import { useApp } from '../store'

const SOURCE = 'https://github.com/vincentboulard/Square-Harness'

const NOTEBOOKS: Partial<Record<Mode, string>> = {
  free: 'Say what you need. The model picks the workflow and its effort and starts at once; its card says what started and can cancel it.',
  prove: 'A solver writes a whole proof, a fresh verifier reviews it, and a concrete objection leads to a repair.',
  literature: 'A report on published results that cites the exact passages it read.',
  referee: 'A review of a manuscript, with line-level citations for every claim it checks.',
  writeup: 'Clean LaTeX from your notes, drafts or PDFs, in your own macros and style.',
}

export function About() {
  const { status } = useApp()
  return (
    <div className="sheet paper">
      <div className="sheet-inner about">
        <div className="entry">
          <span className="in-margin" />
          <section className="notebook-label" aria-label="Square Harness">
            <h1 className="label-title">Square Harness</h1>
            <p className="label-subtitle">Mathematical research with language models you run yourself</p>
            <dl className="label-lines">
              <div><dt>Authors</dt><dd>Vincent Boulard, Louis Carillo</dd></div>
              <div><dt>Version</dt><dd>{status?.version || '…'}</dd></div>
              <div><dt>Licence</dt><dd>Apache License 2.0</dd></div>
              {status && <div><dt>Model</dt><dd>{status.model}</dd></div>}
            </dl>
          </section>
        </div>

        <div className="entry about-letter">
          <span className="in-margin" />
          <div className="prose about-prose">
            <p>
              Square Harness is a workbench for doing mathematics with a language model that you run yourself, on your
              computer or on your lab's server. You write the statement; the harness keeps it fixed, gives the model a
              bounded budget, saves every request and every answer, and shows you exactly what was read and what was
              written. Apart from the literature searches you allow, your work stays on the machines you chose.
            </p>
            <p>
              We wanted a tool that is honest about what a model can and cannot do. A proof the model approves is a
              candidate for you to check, not a certificate. A citation points to the lines that were actually read. When
              the model is unsure, the harness says so instead of smoothing it over.
            </p>
            <p>
              It is young, experimental software. Read what it writes the way you would read a draft from a colleague:
              with interest, and with a pencil in hand.
            </p>
            <p className="signature">Vincent and Louis</p>
          </div>
        </div>

        <div className="entry">
          <span className="in-margin" />
          <h2 className="section-title about-heading">The notebooks</h2>
        </div>
        {RAIL.map((mode) => (
          <div key={mode} className="entry about-notebook">
            <span className="in-margin"><span className={`about-cover mode-${mode}`} aria-hidden="true">{MODES[mode].glyph}</span></span>
            <p><a href={href(mode)} className="about-name">{MODES[mode].label}</a> {NOTEBOOKS[mode]}</p>
          </div>
        ))}

        <div className="entry">
          <span className="in-margin" />
          <h2 className="section-title about-heading">What stays true</h2>
        </div>
        <div className="entry">
          <span className="in-margin" />
          <ul className="about-list">
            <li>The interface loads nothing from the internet. Online search is a switch on each job and conversation, and a launch with <code>--offline</code> locks it off.</li>
            <li>Every job keeps its requests, streamed answers, candidates and reviews in the folder's <code>.mathagent</code> directory.</li>
            <li>Every job has limits on attempts, tokens and time. Resuming continues with what is left and never grants more.</li>
          </ul>
        </div>

        <div className="entry">
          <span className="in-margin" />
          <p className="about-links">
            Source code, documentation and issues: <a href={SOURCE} target="_blank" rel="noopener noreferrer">{SOURCE.replace('https://', '')}</a>.
            {' '}Fonts and libraries in this interface: <a href="/THIRD-PARTY-NOTICES.txt" target="_blank" rel="noopener">open-source notices</a>.
            {' '}The Poincaré effort is named in honour of Henri Poincaré.
          </p>
        </div>
      </div>
    </div>
  )
}
