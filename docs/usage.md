# Square Harness usage guide

Detailed reference for the experimental 0.4 release. Start with the
[README](../README.md) for installation. Commands below assume an existing
workspace at `~/research/my-paper` and your own files inside it.

## GPU servers and parallel proofs

Ollama remains the desktop default. For the two-V100S llama.cpp deployment:

```bash
square-harness --backend llamacpp --host http://localhost:8000 \
  --model square-qwen --workspace ~/research/my-paper \
  --ctx 32768 --predict 8192 --seed 42 \
  --request-timeout 7200 --proof-seconds 14400 --proof-selection-seconds 1800
```

Use `--backend openai` for vLLM. The `llamacpp` adapter uses its native template,
exact prompt-token counting, schema output and reasoning fields. Offline mode
still requires a loopback endpoint; an SSH tunnel can connect to your own GPU
server. The server controls its actual context allocation: `--ctx` is the
harness's input/output budget and must not exceed the server's configured limit.
Use [the OVH guide](ovh.md) for the pinned model and server setup.

Add `--proof-workers 3` to run three independent proof branches on the same
statement. Each branch has its own process, client, workspace and ledger; only
explicitly pinned source files are copied in. They share a fixed total
`--proof-tokens` ceiling, including selection reviews. The default 60,000 tokens
allocate 17,952 per branch and 6,144 for selection. Unused allocations stay unused.
Solver seeds differ between branches; critics and selection reviews use
temperature zero. Separate contexts cannot guarantee independent mathematical mistakes.

The final selector audits whole candidates separately and returns an existing
answer, without combining proofs. No clipped, truncated or malformed review
can approve a candidate. The status is a model judgment, never a proof certificate.
All candidates and the selected `answer.md` remain in the saved portfolio directory.
If there is no eligible complete output, no selected answer is created.

Parallel portfolios are currently one-shot jobs: an interrupted parent is not
automatically replayed. Child proof ledgers retain the existing checkpoint and
budget protections and can be inspected independently. Use `--proof-workers 1`
for the original sequential workflow and `/resume`. Literature tools are not
enabled in parallel portfolios. The [benchmark runner](benchmark.md) provides
isolated datasets, aggregate request limits and independent grading exports.

## Persistent proof work

`/prove` now starts a saved, bounded proof job. Plain questions in the default
`prove` mode do the same. Supply a self-contained goal: new jobs do not inherit
the previous chat's hypotheses. The controller keeps the original goal fixed
while assigning smaller mathematical tasks. Attempts, exact arguments, reviews,
dependencies and objections accumulate in a durable ledger. Full records remain
on disk while later calls receive selected records that fit the working context.
This context rebuilding preserves access to earlier work; it does not establish
the mathematical validity of a claim.

The persistent proof cycle is:

1. **Attempt a focused task.** The solver begins with the `--predict` allowance.
   After output truncation, later attempts can use more tokens, up to
   `--proof-max-predict`, available context space and the remaining job budget.
2. **Write a checkpoint when needed.** A truncated or thinking-only attempt gets
   a separate call with thinking disabled to extract a usable argument and its
   unresolved steps. The original trace is retained. A checkpoint is always
   marked partial, even if its content appears sufficient: a later solver must
   submit the complete, self-contained proof before it can receive a full audit.
3. **Review the current argument.** A fresh critic sees the original task and
   candidate without the ledger's previous judgments. It checks concrete
   inferences, identifies the first unsupported step, and preserves useful
   partial work with its restrictions.
4. **Update the ledger.** A separate recorder receives the candidate, that fresh
   critique, and selected ledger records. It can extract up to four small claims
   or obligations rather than recording the entire theorem as a single gap.
   Its reference view omits historical next-task instructions and places the
   current candidate and critique last, to reduce copying of stale assignments.
   The controller validates the review structure and dependency references,
   preserves the critic's concrete objection even if the recorder omits it,
   and deduplicates identical records. These are consistency checks, not a
   mathematical verifier.
5. **Choose the next task.** After two rounds without recognized progress, a
   planning call proposes a different task or approach. Repeated truncation can
   also trigger an attempt with thinking disabled so the model writes usable
   mathematics. A stalled approach does not by itself terminate the whole goal.
   If the recorder repeats the previous assignment verbatim while the fresh
   critic identifies remaining work, that current obligation takes precedence.
6. **Audit a proposed complete proof.** A separate call checks the full argument
   against the original goal before a `candidate_complete` result is recorded.
   All historical objections remain visible to this audit, including those the
   recorder claims are resolved; the auditor must check the actual repair.

Solver, checkpoint writer, planner, critic, recorder and auditor are separate
calls to the same local model. They share one saved token/time budget, and can share the same
mathematical blind spots. The prompts and scheduling rules are general rather than tied to a particular
theorem. Literature
tools are disabled in proof work by default, including when ordinary research
is online. Use `--proof-literature` to opt into cached literature tools and add
`--online` for fresh searches and downloads. Literature-assisted proof search
changes what the benchmark measures.

To work on a statement in your own workspace:

```bash
square-harness --offline --workspace ~/research/my-paper --ctx 16384 --predict 4096 --proof-max-predict 8192 \
  --proof-rounds 10 --proof-tokens 60000 --proof-seconds 1800 \
  --proof-file statement.tex \
  --prompt "Prove the complete statement in statement.tex using only its hypotheses."
```

`--proof-file` pins the complete source as part of the saved job; paths are
relative to the workspace. Repeat it for multiple source files; use this option
for filenames containing spaces. When no `--proof-file` is supplied, simple file
names in the goal are detected, so `/prove Read statement.tex and prove the
lemma rigorously.` works. The original goal and source snapshots stay unchanged
on resume, even if the manuscript is subsequently edited.

The terminal prints the proof ID, status and debrief path. Proof records are
saved automatically under `<workspace>/.mathagent/proofs/<proof-id>/`:

- `state.json`: versioned ledger, model settings, policy-file hash and persistent budget counters.
- `ledger.md`: readable mathematical progress and open obligations.
- `report.md`: debrief, useful partial results, remaining gaps and suggested next work.
- `artifacts/`: exact inference request payloads, saved arguments, reviews and
  streamed incomplete attempts. Each dispatched call records its request path
  in `state.json`, so you can inspect which context and settings it received.

Save the ID or use `/proofs` to find it. Ctrl+C pauses the current proof; completed
checkpoints survive process exit. Resume with the remaining original budget:

```bash
square-harness --workspace ~/research/my-paper --resume <proof-id>
square-harness --workspace ~/research/my-paper --prompt "/ledger <proof-id>"
square-harness --workspace ~/research/my-paper --prompt "/proofs"
```

Inside the terminal, use `/resume [proof-id]` or `/ledger [proof-id]`; without an
ID they use the latest selected job, or the latest active job if none is selected.
`--resume` without an ID also selects the latest active job. If no active job
exists, the latest job is selected. `/ledger` and `/proofs` work offline.

A resume restores the saved model and inference settings, sources and limits,
including the adaptive output ceiling and whether literature tools were enabled.
Passing `--proof-literature` on resume cannot convert an unaided job into an
assisted one. Use the same Ollama host. Changing
`--proof-rounds`, `--proof-tokens`, `--proof-seconds` or `--proof-max-predict` on
a resume does **not** replace those saved settings or grant another budget.
Exhausted and completed jobs keep their report; a new `/prove` starts a separate
job. Interrupted v0.2 jobs remain readable and resumable. When an older active
job has no saved adaptive ceiling, the controller records a ceiling of at least
8,192 tokens, or its original `--predict` value if larger, before dispatching new
work. A v0.2 job already marked `stalled` remains terminal. Old attempts are not silently imported into new jobs.

The default limits are **10 mathematical rounds, 60,000 generated tokens and
1,800 seconds**. All inference, including checkpoint writing, planning, reviews,
recovery calls and the final audit, consumes the same saved budget. Solver calls
start at `--predict` (default 4,096), and can grow after truncation to
`--proof-max-predict` (default 8,192). The ceiling must be at least `--predict`
for a new proof. It is a maximum, not an allocation guaranteed to fit each call:
solver output is also bounded by half the configured context. The current prompt
must fit the space that remains. After tool use, the controller recalculates
that space with the retained tool calls and returned evidence, dropping optional
ledger records from the working context first; they remain available on disk.
If essential material still does not fit the conservative context estimate,
the controller pauses with `needs_context` instead of silently dropping it.
With the default `--ctx 8192`, the solver can use at most
4,096 output tokens; `--ctx 16384` allows an 8,192-token solver allowance when
the prompt fits. Review and checkpoint calls use their own bounded allowances
and disable thinking. `--no-think` keeps solver thinking disabled; otherwise the
controller can alternate thinking with direct written attempts when it detects
repeated truncation.

`--max-rounds` still controls tool rounds in ordinary chat; it does not set the
number of proof rounds. A proof job ends when it has an audited candidate or
reaches its overall round, token or time limit; repeated lack of progress changes
the task instead of applying v0.2's four-round `stalled` stop. These decisions
are heuristics: they cannot detect every equivalent failed approach or guarantee
that the model discovers a new one. A crash during active
work conservatively charges the interval until resume against elapsed time;
an ordinary Ctrl+C pause does not charge offline time. Network timeouts and
already-running approved Python snippets may delay a stop. If a damaged state
requires backup recovery, automatic inference stops because the newest budget
reservation may be missing; retained artifacts remain available for inspection.

The final status describes **model review, not formal proof verification**.
The same model can repeat an error in both solver and reviewer roles. A completed
candidate needs independent mathematical checking; partial results retain their
objections and review status. The controller does not establish research-level mathematical reliability.

## Offline and literature-assisted modes

The default `--offline` mode disables external search and document downloads.
It still permits inference through a loopback Ollama server and reading local
files or already cached papers. There is no Google Scholar scraping or browser
automation. The network policy is checked in the tool implementation, not just
stated in a prompt.

The controls are separate:

| Launch option | Literature/referee work | Proof finding |
| --- | --- | --- |
| `--offline` (default) | Local manuscripts and cached sources only | No literature tools |
| `--online` | External searches and public document retrieval enabled | No literature tools |
| `--offline --proof-literature` | Local/cached sources only | Cached literature tools enabled explicitly |
| `--online --proof-literature` | External research enabled | Literature tools enabled explicitly |

Disabling literature tools is not an operating-system network sandbox. In
`--online --allow-python` sessions, explicitly approved Python can still use the
network. Use the default offline mode for unaided proof experiments.

Offline mode cannot undo information already included in a saved online job.
Use a fresh offline proof job and a statement-only source for an unaided
benchmark. Inference remains local unless you explicitly choose a remote Ollama
host in online mode. Cached material and source summaries are not assumed true:
the agent must inspect the relevant passage and check its hypotheses.

## Bibliographical and referee reports

Two bundled **Square Harness skills** supply the investigation and reporting
instructions. They are native harness resources, not Codex plugins, and require
no separate skill installation:

- `literature`: define the scope, search and select sources, read relevant
  passages, compare precise results and hypotheses, identify limitations, and
  write a bibliographical report in Markdown.
- `referee`: read a pinned manuscript, map its main claims and proof dependencies,
  investigate concrete objections, check related literature when enabled, and
  write a referee-report draft distinguishing demonstrated errors, missing
  justifications, unresolved concerns and presentation issues.

The controller plans the investigation, runs bounded tool-assisted research
rounds, drafts a report, asks a fresh model context to review it, and revises it
when budget remains. Each research round rebuilds a compact working context from
saved material. Papers and evidence stay on disk; complete PDFs are not repeatedly
inserted into the model's prompt. External content is treated as evidence, never
as instructions to the agent. The search/reading tools are available in ordinary
critic and explore chat as well; those chats do not get a persistent research
workflow automatically.

The report instructions require encountered source identifiers and locators,
such as `[doc-IDENTIFIER:L12-L28]` for external passages and `[M1:L5-L19]` for
a pinned manuscript. An abstract or search hit remains a discovery lead until
the relevant text has been read. A model-written summary is not a substitute
for the source passage or a check of its mathematical assumptions.

Start an online bibliographical report from the project directory:

```bash
source .venv/bin/activate
square-harness --online --workspace ~/research/my-paper --mode literature --ctx 16384 \
  --prompt "Review observability inequalities for strongly continuous semigroups. Compare hypotheses and conclusions, identify the papers actually read, and describe search limitations." \
  --output observability-review.md
```

The output path is relative to the workspace. `--output` exports a copy of the
saved Markdown report; an existing file is not silently overwritten. A report is
also saved inside the job directory when no output path is supplied.

For a referee report, pin the manuscript rather than relying on a reference to a
filename buried in the prompt:

```bash
square-harness --online --workspace ~/research/my-paper --mode referee --ctx 16384 \
  --research-file manuscript.tex \
  --prompt "Assess the main results and arguments, check related literature, and prepare a referee report with precise manuscript and source locations." \
  --output referee-report.md
```

Repeat `--research-file` for additional files; local PDF manuscripts use the
PDF reader described below. The combined pinned text is limited to 2 MB. For a local assessment with no
external research, replace `--online` with `--offline`. An offline report must
state that its literature coverage is limited to the supplied and cached
material. Pinned manuscript snapshots stay fixed on resume, even if the source
file changes. Long manuscripts are read in bounded excerpts with stable
manuscript/line locators instead of being silently cut to fit a prompt.

At the interactive prompt, `/literature <topic>` and `/referee <request>` start
the same saved workflows. Online access is selected at launch; a document or
model response cannot grant it. Use the launch-time `--research-file` option to
pin files for these jobs.

Research jobs live under
`<workspace>/.mathagent/research/<research-id>/`:

- `state.json`: the goal, source snapshots, workflow progress, model settings and
  remaining budget.
- `report.md`: the current Markdown report, including its completion/review status.
- `artifacts/`: retained intermediate work and inference records.

Save the printed research ID. Inspect jobs or resume with their original limits:

```bash
square-harness --workspace ~/research/my-paper --prompt "/researches"
square-harness --workspace ~/research/my-paper --prompt "/research-report <research-id>"
square-harness --online --workspace ~/research/my-paper --research-resume <research-id>
```

Inspection does not require a running model. Ctrl+C pauses a research job;
resume uses its remaining saved budget rather than granting a new one. A job
that exhausts a limit retains the best available partial report and evidence.
`reviewed` means reviewed by a model, not certified correct or ready for journal
submission. Search coverage can be incomplete; absence from results is not
evidence that a result is new.

### Search providers and paper reading

Literature tools use supported APIs and retain results under
`<workspace>/.mathagent/literature/`. `searches/` caches discovery results;
`documents/` stores source metadata and extracted paper text. The same cache can
serve later jobs in that workspace, including offline reading.

| Tool | Purpose |
| --- | --- |
| `search_papers(query, provider, limit)` | Search `arxiv` (default), `semantic_scholar` or `openalex`; default five results |
| `search_web(query, limit)` | General public web discovery through Brave Search |
| `open_paper(identifier_or_url)` | Retrieve/cache a paper and return its document identifier and reading information |
| `read_paper(document_id, start_line, end_line)` | Read a bounded, line-numbered passage |
| `search_paper(document_id, query)` | Find a term within a cached paper |

Search results provide discovery metadata; full-text reading is a separate step.
A passage is bounded to 80 lines and 6,000 characters. Source extraction can
damage formulas, especially in scanned PDFs or complex layouts. The model is
instructed to flag uncertain notation and avoid presenting an extraction as a
verified mathematical statement. Paywalls and unavailable full text remain
explicit limitations; the harness does not bypass them.

arXiv needs no key. Semantic Scholar and OpenAlex accept optional keys; their
unkeyed access may be limited or rate-limited. General web search requires a Brave
Search API key. Set keys in the terminal before launching, when you need those
providers:

```bash
export SEMANTIC_SCHOLAR_API_KEY="your-key"
export OPENALEX_API_KEY="your-key"
export BRAVE_SEARCH_API_KEY="your-key"
```

These are optional environment variables, not keys to place in the prompt or in
a manuscript. Without any key, start with arXiv. Search provider availability and
quotas are separate from the harness's saved request budget. Provider errors do
not justify fabricated results.

For PDF extraction, the harness first uses `pdftotext` when installed, otherwise
it can use the optional Python package:

```bash
python -m pip install -e '.[research]'
```

On Fedora, `sudo dnf install poppler-utils` supplies `pdftotext`. HTML and text
reading do not require that extra. Local manuscript files can be read without
uploading them; online requests are used only for the selected external search
or document URL.

Provider references:
[arXiv API manual](https://info.arxiv.org/help/api/user-manual.html),
[Semantic Scholar API](https://api.semanticscholar.org/api-docs/),
[OpenAlex API](https://help.openalex.org/api/), and
[Brave Search API](https://api-dashboard.search.brave.com/documentation/quickstart).

### Research budgets and confidentiality

Defaults for one saved research job are:

| Option | Default | What it bounds |
| --- | --- | --- |
| `--research-rounds` | `6` | Investigation rounds |
| `--research-tokens` | `24000` | Generated model tokens across the workflow |
| `--research-input-tokens` | `100000` | Cumulative input-token allowance |
| `--research-seconds` | `900` | Active elapsed time |
| `--research-requests` | `12` | External retrieval request allowance |
| `--research-chars` | `30000` | Returned research evidence character allowance |

Planning, drafting, review and revision share the model budget. Input is conservatively estimated before each call and corrected when Ollama
reports actual usage. Repeated context counts again: a small output does not
make a long paper free to process. Bounded search results, cached downloads, short
located excerpts and source reuse reduce waste. Limits control the amount of
work, not its scholarly completeness; increase them deliberately for a larger
review. Context size remains a separate per-call constraint.

Online queries and requested URLs leave your machine even while model inference
runs locally. A referee workflow should use generic public subject queries,
keeping unpublished manuscript text local. Review the relevant journal's policy
and your confidentiality obligations before allowing any external research;
the harness cannot determine whether a particular disclosure is permitted.
Offline mode is the option when external requests must be excluded. The tool
policy and skill instructions reduce disclosure risks but do not make model
judgment a confidentiality guarantee.

## Commands

| Command | Action |
| --- | --- |
| `/prove [goal]` | Start a persistent proof job, or switch to proof mode |
| `/critic [question]` | Look for gaps and counterexamples |
| `/explore [question]` | Explore approaches, with labelled heuristics |
| `/literature [question]` | Start a saved bibliographical investigation and Markdown report |
| `/referee [question]` | Start a saved manuscript assessment with literature checking when enabled |
| `/review` | Fresh-context critique of the last answer |
| `/resume [proof-id]` | Continue a saved proof using its remaining budget |
| `/ledger [proof-id]` | Read the saved debrief without contacting Ollama |
| `/proofs` | List saved proof jobs in this workspace |
| `/researches` | List saved literature and referee jobs |
| `/skills` | Show the bundled literature/referee skill paths |
| `/research-report <research-id>` | Display a saved Markdown research report |
| `/research-resume <research-id>` | Resume a saved research job with its remaining budget |
| `/think on` / `/think off` | Change reasoning for new proof jobs and chat requests |
| `/context 16384` | Change context for new proof jobs and chat requests |
| `/status` | Show settings and latest request token counts/stop reason |
| `/clear` | Clear chat history; saved proof jobs remain |
| `/save session.json` | Export settings and conversation inside the workspace |
| `/paste` | Enter multiple lines; submit with `/end` on a separate line |
| `/help`, `/quit` | Help or exit |

Ctrl+C pauses persistent proof or research work. In ordinary chat, it cancels generation and
discards the unfinished conversation turn; writes or computations you already
approved are not undone. Ordinary chat state stays in memory. `/save` exports its
JSON transcript, but chat transcript import is not implemented. Proof and research jobs are
saved and resumed separately.

One-shot use:

```bash
square-harness --workspace ~/research/my-paper --mode critic --prompt "Read manuscript.tex and check the proof."
```

## Tools and permissions

- `list_files`: supported text files, at most 2,000 files scanned.
- `read_file`: numbered excerpts, at most 200 lines and 6,000 result characters.
- `search_text`: case-insensitive literal search, at most 50 matches.
- `write_file`: creates/replaces text only after displaying a diff and receiving `y`.
- `run_python`: opt-in and approved separately for each snippet.

Text tools accept `.tex`, `.bib`, `.md`, `.txt`, `.py`, `.json`, `.yaml`, `.yml`,
`.toml`, and `.csv`. Hidden paths, paths outside the workspace, and files over
1 MB are excluded. Recursive listing/search does not follow symlinks. These text
tools are intended for a normal user-owned research directory, not hostile
concurrent filesystem changes. Literature tools add bounded PDF/HTML/text reading
and optional online search, as described above. LaTeX compilation is not implemented.

To enable Python/SymPy computations:

```bash
python -m pip install -e '.[math]'
square-harness --online --workspace ~/research/my-paper --allow-python
```

Each snippet is shown before execution and requires `y`. **Python is not
sandboxed:** it can access your account's files and network regardless of the
text-tool workspace boundary. Run only snippets you have inspected. Linux process
timeouts (30 seconds), CPU limits and output limits are resource guardrails, not
security isolation. Python tools use the same interpreter/environment as the agent.

By default, model requests go only to `http://localhost:11434`. Offline mode
rejects a non-loopback Ollama host and `--allow-python`: arbitrary Python could
bypass the network restriction. `--online` permits an explicitly chosen remote
`--host`; prompts and file excerpts will then go to that server. Online mode is
also required for the optional Python tool, even if your particular snippet uses
no network. There is no telemetry. Persistent proof jobs automatically save source
snapshots, model outputs and progress locally. Ordinary chat transcripts are
saved only through `/save`. These records can contain manuscript excerpts.
`.mathagent/` is ignored by Git so saved jobs are not accidentally committed.

## Memory and generation settings

Defaults are `--ctx 8192 --predict 4096`. The output budget includes thinking.
For a larger proof working context, if your machine can accommodate the memory:

```bash
square-harness --ctx 16384 --workspace ~/research/my-paper
```

Larger context can increase memory use and CPU offload. Check `ollama ps` in
another terminal to see the actual allocation. This harness does not tune your GPU.
`--predict` sets the ordinary-chat output limit and the initial proof solver
allowance. In proof mode, `--proof-max-predict` sets the adaptive ceiling; review
and other controller calls have separate smaller allowances. Increasing either
output setting does not enlarge the context allocation.

Thinking is enabled by default, with a live character-count status; the model's
thinking text is hidden unless you pass `--show-thinking`. This changes display,
not the amount of computation. `/think off` changes new solver and chat requests.
Proof checkpoint writing and review disable thinking independently, and the
controller may use direct written attempts after repeated solver truncation.

Ollama stop reasons are reported when generation hits an output limit. A stop
is **not automatically evidence of context exhaustion**. Persistent proof work
retains incomplete output and uses bounded recovery/review steps before deciding
what to try next. In ordinary chat, a larger output/context budget or disabled
thinking can help a simple smoke test. Check `/status` for the most recent
ordinary-chat `done_reason` and token counts; proof counters are in the saved job.

Ordinary chat uses a conservative **byte-based estimate, not an exact tokenizer**,
to budget input. It announces when it removes the oldest complete user turn,
keeping tool calls paired with their results. If the current turn alone is too
large, it stops with guidance rather than silently cutting a manuscript excerpt.
The estimate cannot guarantee that Ollama itself never truncates context. Read
small excerpts and restate essential hypotheses after long conversations.

In ordinary chat, there are at most eight tool rounds per user query, followed by a final call
without tools (`--max-rounds` changes this). Tool results are bounded and marked
when truncated. Thinking is retained through the current tool loop, then removed
from stored chat history. No model-specific `preserve_thinking` or
`reasoning_effort` extension is assumed.

## What is inside

| File | Responsibility |
| --- | --- |
| `mathagent/cli.py` | Terminal interaction, commands, approval UI |
| `mathagent/agent.py` | Streaming HTTP and bounded tool loop |
| `mathagent/proof.py` | Bounded proof tasks, model review, context rebuilding and debriefs |
| `mathagent/proof_policy.py` | General mathematical role prompts and structured response schemas |
| `mathagent/ledger.py` | Durable proof state, budgets, arguments and reports |
| `mathagent/tools.py` | Workspace tools and opt-in Python |
| `mathagent/prompts.py` | Mathematical policy and modes |
| `mathagent/literature.py` | Search APIs, paper extraction, cache and network policy |
| `mathagent/research.py` | Bounded research jobs, source snapshots and Markdown reports |
| `mathagent/skills/*/SKILL.md` | Native literature and referee workflow instructions |
| `tests/test_agent.py` | Protocol, control-flow and permission tests |

The implementation calls Ollama's native HTTP API directly, so no LLM framework
or Ollama Python SDK is needed. Rich and prompt-toolkit are just UI dependencies.

Official protocol references:
- [Ollama chat API](https://docs.ollama.com/api/chat)
- [Ollama tool calling and streaming](https://docs.ollama.com/capabilities/tool-calling)
- [Qwen3.8 27B model tag](https://ollama.com/library/qwen3.8:27b)

## Verification and limitations

Run the tests without downloading weights:

```bash
python -m unittest discover -s tests -v
```

The test suite covers workspace escapes, approval-denied writes and Python,
stream interruption, context budgeting through tool continuations, output-limit reporting, durable proof
state, persistent budgets, adaptive output limits, partial checkpoints, blind
critique, atomic claim recording, stale-task handling, CLI routing and offline inspection, and
streaming HTTP against a simulated Ollama server. Research tests exercise
offline access, provider parsing, cached paper evidence, request/input budgets,
report lifecycle and CLI routing with controlled fixtures. These tests do
not measure mathematical quality or verify model-specific tool-calling behavior.
The program remains a prototype with model-guided proof search
and review, not a formal proof checker. Literature retrieval does not establish
that a cited theorem applies, that a paper is novel, or that a manuscript is
correct. Persistent ordinary-chat sessions and mathematical correctness
guarantees are not implemented.
