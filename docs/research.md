# Optional chat and research workflows

These existing tools remain available alongside the focused
[proof workflow](usage.md). They do not participate in proof-mode benchmarks.

## Manuscript chat

```bash
square-harness --workspace ~/research/my-paper --mode critic \
  --prompt "Read manuscript.tex and identify the first unsupported implication."
```

Use `/explore` for approaches and heuristics, `/critic` for objections, and
`/review` for a fresh-context critique of the last chat answer. Chat can list,
read and search supported text files in the workspace. Writes display a diff
and require approval. Text excerpts are bounded; the model must not treat an
excerpt as the whole manuscript.

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

For a manuscript review:

```bash
square-harness --offline --workspace ~/research/my-paper --mode referee \
  --research-file manuscript.tex \
  --prompt "Assess the arguments and distinguish demonstrated errors, gaps and unresolved concerns." \
  --output referee-report.md
```

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

Search tools support arXiv, Semantic Scholar and OpenAlex; public web discovery
uses Brave Search. Optional keys are `SEMANTIC_SCHOLAR_API_KEY`,
`OPENALEX_API_KEY` and `BRAVE_SEARCH_API_KEY`. Do not place keys in prompts or
manuscripts. No key is required to begin with arXiv; service limits still apply.
Cached sources live under `.mathagent/literature/`.

## Saved reports and limits

Reports live under `.mathagent/research/<id>/`, with state, `report.md` and
artifacts. Use `/researches`, `/research-report <id>` and `/research-resume <id>`
to inspect or continue. Inspection needs no running model. Ctrl+C saves progress;
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
inspect any snippet before approving it. LaTeX compilation is not provided.
