# Optional chat, research and write-up workflows

These existing tools remain available alongside the focused
[proof workflow](usage.md). They do not participate in proof-mode benchmarks.

## Manuscript chat

```bash
square-harness --workspace ~/research/my-paper --mode critic \
  --prompt "Read manuscript.tex and identify the first unsupported implication."
```

Use `/explore` for approaches and heuristics, `/critic` for objections, and
`/review` for a fresh-context critique of the last chat answer. Chat can list,
read and search supported text files in the workspace (including `.sty` and
`.cls`); `read_file` also returns the locally extracted text of workspace PDFs.
Writes display a diff and require approval. Text excerpts are bounded; the model
must not treat an excerpt as the whole manuscript.

`--predict` bounds chat output including thinking; `--max-rounds` bounds tool
rounds. `/think on|off` changes chat reasoning, and `/context N` changes the
context allocation for new work. `/clear` clears chat history; `/save session.json`
exports a transcript. Chat transcript import is not implemented.

## Literature and referee reports

Install PDF support when needed:

```bash
python -m pip install -e '.[research]'
```

For an online literature investigation:

```bash
square-harness --online --workspace ~/research/my-paper --mode literature \
  --prompt "Compare observability inequalities for strongly continuous semigroups. State the hypotheses and identify passages actually read." \
  --output literature-review.md
```

For a manuscript review, see [Reviews](#reviews-quick-check-journal-report-explanation) below.

Repeat `--research-file` for additional sources. These files are pinned on job
creation. `--output` is relative to the workspace and does not silently overwrite
an existing file. Without it, the report remains in the saved job directory.
An offline report can use local and cached sources only. External research
requires `--online`; model output cannot grant network access.

The bundled `literature` and `referee` skills guide bounded investigation,
report drafting and model review. They require source identifiers and passage
locations. Search hits and abstracts are leads, not evidence that a theorem
supports the requested assertion. PDF extraction can corrupt formulas; inspect
important passages in the original source.

Search tools support arXiv, Semantic Scholar and zbMATH; reading lists and
literature checks also use Crossref and the OpenCitations index. Public web
discovery uses Brave Search. Optional keys are `SEMANTIC_SCHOLAR_API_KEY` and
`BRAVE_SEARCH_API_KEY`. Do not place keys in prompts or
manuscripts. No key is required to begin with arXiv; service limits still apply.
Cached sources live under `.mathagent/literature/`.

## Reviews: quick check, journal report, explanation

Review comes in four kinds, on one page of the interface (and as `/quick_review`,
`/referee`, `/detailed_review` and `/explain` in the terminal):

- **Quick check** (`quick_review`): is this proof correct? Paste the statement and its
  proof, or pin a file and name the result (`Lemma 3.2`, a label such as `lem:compact`).
  Independent verifier passes read the proof through different lenses (line by line,
  adversarial, results used, overall structure); each alleged error is re-checked by a
  fresh call; the verdict is *no issue found*, *issues found* or *uncertain*, and every
  objection cites the proof lines with an exact quote. The proof is never rewritten; each
  issue comes with a suggested repair.
- **Review** (`referee`): a referee report on a pinned manuscript for understanding and
  presentation. The whole paper is read part by part: an overview (question, main results,
  proof strategy, how the sections fit, assumptions), part-by-part notes, and typos and
  presentation problems with corrections, then an assessment, a recommendation and a note
  to the editor. The proofs are not checked. From a PDF, only comments on words and
  sentences are kept, since its formulas are garbled; a TeX source gives the full check.
- **Detailed review** (`detailed_review`): the review, then the proofs are checked one at a
  time, main results first, as far as the effort allows, with twice the budget of a review.
  The controller maps the manuscript (sections, numbered statements, their proofs, the
  results and equations each proof uses, its citations). Numbered major and minor comments
  on the proofs come from checked findings, with questions to the authors and a table of
  what was checked and what was not; when a main result's proof was not checked, the
  recommendation says it is provisional. Online, results the proofs cite are checked with
  the literature check and the contribution is compared with search results (titles and
  abstracts only).
- **Explanation** (`explain`): a precise explanation of a result and its proof for a
  mathematician from another field: what it says, the idea, a sentence-by-sentence
  walkthrough (every line of a short proof is covered), the standard facts used, and
  where each hypothesis matters. An accuracy pass checks it against the source.

```bash
square-harness --offline --workspace ~/research/my-paper --mode referee \
  --research-file manuscript.tex --prompt "Referee this manuscript." --output referee-report.md
square-harness --offline --workspace ~/research/my-paper --mode detailed_review \
  --research-file manuscript.tex --prompt "Check the proofs too." --output detailed-report.md
square-harness --offline --workspace ~/research/my-paper --mode quick_review \
  --research-file manuscript.tex --prompt "Lemma 3.2"
```

Effort sets the depth: the number of verifier passes for a quick check (1 at Low to 5 at
Poincaré, with re-checks from Medium), and for a detailed review how far down the
dependency graph the checks go (Low: main results; Medium: and what their proofs use;
High: every proof in the body; Extra high and Poincaré: appendices too, with two passes
on the main results). Each model call sees one proof with only what it needs, so small
contexts work; on a vLLM server (`--backend openai`) independent checks run
`--assistant-concurrency` at a time. Checks think before they answer when the output
ceiling leaves room. The manuscript map reads TeX best; for PDFs it relies on the text
extraction, whose formulas can be garbled.

## LaTeX write-ups

The write-up workflow turns rough notes, a draft or a PDF into a LaTeX document in
your own style. It is exposition, not research: the model must keep every
statement, hypothesis and constant from the notes, add no results, and mark
unclear points with `% TODO:` comments for you to settle.

```bash
square-harness --workspace ~/research/my-paper --mode writeup \
  --research-file notes.md --template-file mymacros.sty --template-file template.tex \
  --output lemma.tex --prompt "A short section stating the lemma and its proof. Keep my notation."
```

Template files (`.sty`, `.cls`, `.tex`, `.bib`) are pinned like the notes. The
controller reads them in full and gives the model their macro definitions and
theorem environments to use; a template `.tex` contributes its preamble. The
model plans sections from the numbered note lines, then writes one section per
call from exactly the lines that section cites, so long notes stay within the
context. Before every paragraph it writes a comment such as `% src: [M1:L12-L30]`;
the controller checks those against the lines actually read and reports
paragraphs without one. When `latexmk` is installed, the document is compiled in
a temporary folder without shell escape and with TeX's paranoid file access
(`openin_any=p`, `openout_any=p`); sections named in LaTeX errors get one repair
round. A fresh model review compares the result with the notes.

The `.tex` file (and the PDF when it compiled) is written to the workspace under
the `--output` name, or `<notes>-writeup.tex`; an existing file is never
replaced. A job is `complete` only if it compiled, every paragraph is traced to
read lines and the review finished; otherwise it is `partial`, with the reasons in
its report. Resume and budgets work as for literature reports. In the interface,
template files can be remembered under a name and reused in any folder.

## Saved reports and limits

Reports live under `.mathagent/research/<id>/`, with state, `report.md` and
artifacts; write-ups are saved there too. Use `/researches`, `/research-report <id>`
and `/research-resume <id>` to inspect or continue. Inspection needs no running model. Ctrl+C saves progress;
resume uses the original remaining allowance.

| Option | Default |
| --- | ---: |
| `--research-rounds` | 6 |
| `--research-tokens` | 24,000 generated tokens |
| `--research-input-tokens` | 100,000 cumulative input tokens |
| `--research-seconds` | 900 seconds |
| `--research-requests` | 12 external retrieval requests |
| `--research-chars` | 30,000 returned evidence characters |

Planning, research, drafting and review share these limits. A model-reviewed
report is not a verified proof or an exhaustive literature survey. Failure to
find a result does not establish novelty.

## Permissions

Online queries and requested URLs leave the computer. An explicitly chosen
remote model server receives prompts and file excerpts. Keep unpublished or
confidential material out of external queries.

The optional `--allow-python` tool requires `--online` and per-snippet approval.
It runs as your account and is **not sandboxed**; workspace file-tool boundaries
do not restrict arbitrary Python. Install `.[math]` for NumPy/SymPy support and
inspect any snippet before approving it. Only the write-up workflow compiles
LaTeX, in the restricted `latexmk` run described above.
