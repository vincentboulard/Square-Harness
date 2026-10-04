# Usage

Proof mode is the terminal default. It solves one problem, reviews the full candidate,
and repairs a concrete objection when enough budget remains. It has two model
roles—solver and verifier—and ordinary code for control and storage.

## Start a proof

Install from the [README](../README.md), start your model server, and use:

```bash
square-harness --workspace ~/research/my-paper \
  --proof-file statement.tex --prompt "Prove the statement in statement.tex."
```

`--proof-file` pins a complete UTF-8 source in the saved job. Paths are relative
to the workspace; repeat the option for additional files. Put the actual problem
and necessary hypotheses in these files, not references to unprovided documents.
Without files, give the complete problem in the prompt. A file too large for the
model context needs a smaller, self-contained problem statement; the controller
must not silently omit hypotheses.

An interactive session provides `/prove <problem>`, `/proofs`, `/proof-report <id>`
and `/resume <id>`. `/paste` accepts multiple lines, ending with `/end`.

## What happens

1. **Solve.** One substantial, tool-free call asks for a complete proof. Thinking
   is enabled. The problem and pinned sources are retained exactly.
2. **Verify.** A fresh context receives the solver's task (quoted as reference) and
   the entire written candidate. It returns a verdict, an explanation and located
   objections. The verifier samples with the solver temperature by default
   (`--proof-verify-temperature`); greedy decoding tends to loop on thinking models.
3. **Stop or repair.** If the verifier finds no issue, return that same candidate
   unchanged. Concrete objections are passed to the solver as claims to assess:
   it can repair the proof or rebut a mistaken objection. A revised complete proof
   is reviewed again. An uncertain verdict stops with the candidate retained; it
   does not automatically request a rewrite.

One review decides: a single "no issue found" ends the job, and an uncertain
verdict stops it. Extra tokens are spent only on repairing a concrete objection;
there are no confirmation reviews.

An unfinished response is retained as partial work. Continuing it is another
model request grounded in saved text, not a guaranteed continuation of the prior
request's hidden reasoning. The written text is always included; the saved notes
are included in full or, when they do not fit, as a clearly marked final excerpt.
If even the written text leaves too little room, the original problem is solved
afresh instead. Thinking stays on; the controller does not force a new approach
after an arbitrary number of rounds.

A malformed or truncated review is retried at most once on the same candidate
when budget allows. If still unavailable, the job stops with the candidate
retained. This is an operational failure, not a demonstrated mathematical error.
Review uncertainty must remain visible. Model endorsement,
including agreement after repair, is not a formal proof certificate.

## Budgets

| Option | Default | Meaning |
| --- | ---: | --- |
| `--ctx` | 40,960 | Per-request context, including input and generated output |
| `--proof-solve-tokens` | 32,768 | Maximum generated tokens for a solve or repair |
| `--proof-verify-tokens` | 16,384 | Maximum generated tokens for one verification |
| `--proof-tokens` | 120,000 | Generated-token ceiling for the entire job |
| `--proof-rounds` | 3 | Maximum solver attempts, including the first |
| `--proof-seconds` | 1,800 | Active job time allowance in seconds |
| `--proof-repair-tokens` | solve tokens | Maximum generated tokens for each repair or later attempt |
| `--proof-min-solve-tokens` | min(16,384, solve) | Smallest context-fitted repair/continuation allowance |
| `--proof-verify-temperature` | `--temperature` | Verifier sampling temperature |
| `--request-timeout` | 1,800 | Maximum seconds for a single request |

Output ceilings **include thinking**. A 120k job allowance spans several requests;
it does not require a 120k context. Limits are ceilings, not spending targets.
The controller reserves room for the next useful operation and its review,
and stops when that operation cannot fit in the remaining budget. After the first
solve and review, their measured duration must also fit in the remaining time
before another attempt starts. With the defaults this usually allows one repair.

These are starting settings, not experimentally established optima. A long
candidate plus a long verifier response must fit within the same context window.
The first solve keeps its full allowance, exactly like direct inference. A repair
or continuation may use a smaller output allowance so that its full input fits,
but never less than `--proof-min-solve-tokens`; below that it is an explicit
context failure. Prompts are never shortened invisibly.
The llama.cpp adapter and supported vLLM `/tokenize` endpoints provide exact input
counts; otherwise a conservative byte estimate can reject a request that would fit.
Choose budgets to suit the server and keep enough room for both the problem and
responses. For a smaller initial software check, for example:

```bash
square-harness --workspace ~/research/smoke --ctx 16384 \
  --proof-solve-tokens 4096 --proof-verify-tokens 4096 \
  --proof-tokens 16000 --proof-rounds 2 \
  --prompt "Prove that the sum of two even integers is even."
```

This checks operation, not mathematical quality. A smaller allowance changes the
experiment and may be insufficient for a harder problem.

`--predict` controls ordinary chat, not proof solving. `--no-think` and `/think off`
apply to chat; proof mode always requests thinking. `--show-thinking` changes only
the terminal display. Backend support for reasoning must be verified separately.

## Answers and resume

Every proof has its own directory:

```text
<workspace>/.mathagent/proofs/<proof-id>/
```

The saved state and request artifacts retain the problem, model configuration,
token usage, initial answer, revisions and reviews. `proof.md` is the selected
candidate; read its accompanying report for review status and unresolved issues.
An exported candidate may be unreviewed or incomplete. A normally ended model
response is not necessarily a mathematically complete proof.

The first answer remains available even when a revision is selected. Later
partial output must not replace a complete response automatically. The controller
tracks evidence and versions; it cannot know which proof is mathematically best.
Always report the selected answer separately from other retained candidates.

```bash
square-harness --workspace ~/research/my-paper --prompt "/proofs"
square-harness --workspace ~/research/my-paper --prompt "/proof-report <proof-id>"
square-harness --workspace ~/research/my-paper --resume <proof-id>
```

Inspection works without a running model. Ctrl+C preserves completed artifacts
and recorded usage. Resume keeps the original settings and remaining budgets;
it does not grant a new allowance or change the pinned problem. Use the original
`--backend` and `--host` when they differ from the defaults. Interrupted
requests with uncertain usage must not become free retries.

## Model servers

Ollama is the default (`http://localhost:11434`). For another installed model:

```bash
square-harness --workspace ~/research/my-paper --model <installed-model-tag>
```

For llama.cpp, select its adapter and served model alias:

```bash
square-harness --workspace ~/research/my-paper \
  --backend llamacpp --host http://127.0.0.1:8000 --model square-qwen
```

For an OpenAI-compatible local server, including HyperQwen:

```bash
square-harness --workspace ~/research/my-paper \
  --backend openai --host http://127.0.0.1:18020 --model qwen3.8-27b
```

Match `--ctx` to the actual server capacity. An adapter's existence is not evidence
that every model/server supports thinking and structured verification correctly.
See the [HyperQwen profile](hyperqwen-a10.md) for its explicit reasoning-budget
settings and [benchmark guide](benchmark.md) for reproducible run records.

Offline mode is the default: model requests must use a loopback address. A remote
server requires `--online`, and receives the supplied prompts and source text.
A tunnel to the rented server can keep the client endpoint on loopback. Proof
mode itself offers no web, file-reading or computation tools during inference;
pinned files are loaded by the controller before the job.

## Upgrading from v0.4

Version 0.5 replaces the previous multi-role proof engine. There are no proof
portfolios, cooperative workers, advisor, model recorder, planner, separate audit
stage or checkpoint-summary model call. `/ledger` remains a read-only alias for
`/proof-report`.

Old saved jobs remain inspectable, but cannot resume under the new engine and are
not silently converted. Keep their source snapshot if you need the old execution
behaviour. Start a new job to test v0.5; do not relabel earlier results as v0.5.

Use `--proof-solve-tokens` and `--proof-verify-tokens` instead of the old
`--proof-max-predict`, worker, strategy and selection flags. The simplified
benchmark uses only `raw-single` and `proof`; its protocol is deliberately new.

## Other modes and privacy

`/critic` and `/explore` are ordinary mathematical chat, with workspace tools.
`/review` reviews the last chat answer in a fresh model context. Optional literature
and referee reports and LaTeX write-ups are described in [research.md](research.md);
they are not part of the proof benchmark. The [visual interface](#visual-interface)
offers all of these modes, and proof mode, in a browser.

Manuscript writes require approval. Optional Python execution requires separate
approval and is not sandboxed. Saved jobs can contain unpublished text; keep
research workspaces private. The repository ignores `.mathagent/`, but an external
backup or manual file copy can still include it.

## Visual interface

`square-harness --gui` serves a local web app instead of the terminal prompt.
It uses the same engine, launch options and saved jobs as the terminal: a job
started in one appears in the other, and a job running in a terminal can be
watched live in the browser.

```bash
square-harness --gui --workspace ~/research/my-paper
```

The launch options apply as in the terminal: the model server (`--backend`,
`--host`, `--model`), the sampling options (`--seed`, `--temperature`, `--top-p`),
`--request-timeout`, `--ctx` and the `--proof-*` limits described in
[Budgets](#budgets).

The browser opens automatically; `--no-browser` prevents it and `--gui-port`
changes the port (default 8765, or the next free one). The left rail holds one
notebook per mode, and the list beside it shows that mode's saved jobs or
conversations. The logo above the notebooks opens an About page.

| Mode | What the page shows |
| --- | --- |
| Assistant | A mathematical conversation with direct answers and focused tasks when needed. The assistant reads their results and decides how to continue, within a shared budget (see below) |
| Prove | The selected answer and its review status; each attempt with its written answer (and thinking), the verifier's verdict, explanation and located issues, and the exact model calls; live model output while a call runs; a side panel with budgets and every retained candidate |
| Literature, Review | The report with clickable citations that open the exact passage read, the controller's citation checks, manuscript coverage, evidence, plan, notes and budgets. Review is the interface's name for a referee report |
| Write-up | The LaTeX document with source marks that open the note lines each paragraph came from, compile errors, the PDF, TODOs and template macros (see [LaTeX write-ups](research.md#latex-write-ups)) |

On a wide screen, the job list and a job's side panel (budgets, candidates,
citation checks) can slide away to a thin spine: use the panel icon at the top
of each, and click the spine to bring it back. The choice is remembered by the
browser; when the side panel is hidden, its content is a tab of the page.

New proofs, reports and write-ups take an **effort**. Each level is a fixed budget:

| Effort | Attempts or rounds | Time | Generated tokens | Input tokens (reports) | Web requests |
| --- | --- | --- | --- | --- | --- |
| Low | 1 | 1 min | 30,000 | 120,000 | 4 |
| Medium | 3 | 15 min | 60,000 | 240,000 | 12 |
| High | 5 | 30 min | 100,000 | 400,000 | 24 |
| Extra high | 7 | 1 h | 150,000 | 600,000 | 40 |
| Poincaré | 10 | 2 h | 200,000 | 800,000 | 60 |

A proof attempt reserves one solve and one review (`--proof-solve-tokens` and
`--proof-verify-tokens`); when a level's tokens cannot hold both, the form lowers
these two ceilings in proportion, so every level can make at least one full
attempt. The attempts are a maximum: a job also stops when its tokens or time
run out. Write-ups have no rounds. The summary under the slider states the exact
limits, and Advanced limits lets you edit them, including a proof's solve and
review output ceilings and its context. The Poincaré effort, named after Henri
Poincaré, can run for two hours; pause it whenever you like. The `--proof-*` and
`--research-*` budget flags keep applying to the terminal and to the API.

A proof started in the interface is the same job as `/prove`: the model sees the
statement and the pinned files only, with thinking on and no tools. The form warns
when the statement names a workspace file that is not pinned. Proof jobs made by
v0.4 stay listed with their report, sources and files, but cannot be resumed.

Every job page also lists its saved files: exact request payloads, streamed
answers and intermediate results. Pause stops model work at the next checkpoint,
like Ctrl+C in the terminal; Resume continues with the remaining saved budget.
Standalone jobs use one active job slot. An Assistant turn can include several
focused tasks within that slot; independent tasks can run concurrently on an
OpenAI-compatible server, within the turn's shared allowance.
A job started while another runs, from a form, a suggestion or a Resume button,
waits in the queue shown under the running task, where it can be cancelled; it
starts when the model is free. Approval dialogs replace the terminal's `[y/N]`
prompt for file writes and, with `--allow-python`, for Python.

Conversations are saved in `.mathagent/chats/` so that they survive a reload.
Assistant turns also retain their pending work and budget for resume. In ordinary
critique and exploration chat, a turn that is paused or fails never enters the
model's context; the interface keeps it visible and marks it as discarded. Critique and
exploration conversations made by earlier versions of the interface stay listed
in the Assistant notebook and can be continued.

### Folders and file access

Click the folder name at the bottom of the sidebar to open another folder. Only
folders at or below the interface root can be opened: by default the root is the
`--workspace` folder itself; launch with `--gui-root ~/research` to move between
all your projects. Hidden folders are never shown. Saved jobs and conversations
belong to their folder, so switching waits until the running job is paused.

The line below the folder name says which files the model may **discover and
read by itself** in this folder: LaTeX (`.tex .bib .sty .cls`), PDF (text
extracted locally), Python and other text files. All are allowed by default;
untick a type to hide it from `list_files`, `search_text` and `read_file`. The
choice is saved in the folder's `.mathagent/gui-settings.json` and applies to new
work. Files you pin or drop into a job are always available to that job. Proof
jobs never read files themselves; they receive the pinned text only.

### Dropping files

Drag files from your file manager onto the interface: `.tex .sty .cls .bib .md
.txt .pdf .py` files up to 20 MB are saved in the open folder. A file with the
same name is never replaced; the new one is renamed, for example `notes-2.tex`.
What happens next depends on the page: a proof or report form pins the files, the
write-up form puts `.sty` and `.cls` files in the template slot and the rest in
the notes, and a conversation attaches them to your next message.

### Assistant mode

To remove a saved conversation from the left list, use its trash button and
confirm the displayed title. This deletes the conversation history and its
Assistant checkpoint. Saved proofs, reports and workspace files remain
available. A running conversation must finish or be paused before deletion;
cancel its queued work first if it is waiting for the model.

The ∀ notebook, previously called Default, is a mathematical assistant. It can
answer directly or ask an existing mode to handle a focused task: prove a lemma,
critique an inference, explore approaches, find literature, review a manuscript,
or write up notes. After a task returns, the assistant sees the result and its
unresolved issues and decides whether to answer, ask for clarification, or obtain
more focused help. Delegation is optional; an elementary question can finish with
one short answer.

To find a precise reference for a known result ("where is the Rellich theorem
proved?", "is Theorem 9.26 of Brezis about this?"), the assistant delegates a
**literature check**: a quick lookup that recalls likely sources and checks each
one against zbMATH, Crossref, the OpenCitations index and open papers. Each
reference is graded read, cited, located, contradicted or not found, from the
passages actually read; its card is marked [1] and its log is saved in
`.mathagent/checks/`. A check sees only the delegated statement, never the
conversation or workspace files, so its queries stay public words. A request for
a reading list or survey of a topic goes to the Literature worker instead, which
builds the same verified reading list as the Literature notebook.

For a simple web lookup, the assistant can open a supplied public HTTPS URL,
read page lines and links, and paginate its detected bibliography directly.
It does not need a literature report for these requests. Bibliography entries
without links are retained, and pagination and extraction coverage are reported
separately. Retrieved pages are reference data, not instructions. Opening a
supplied URL needs no search key; general web search still needs the configured
search provider. Uncached pages cannot be fetched when online access is off.

Each delegated task has a card with its request, effort and returned status.
Proofs and reports open in their own notebooks, where their live output, sources,
reviews and artifacts can be inspected. The conversation displays the assistant's
own response. A returned task does not by itself mean that its mathematical
argument is correct. In particular, direct answers receive no automatic proof
review, and a model review remains a model opinion.

Research, review and write-up workers receive the delegated objective separately
from an exact saved copy of the conversation and its source context (a reading
list uses it to set its scope). The
20,000-character objective limit therefore does not apply to a long prior
report. Context must still fit the model window; essential reference context is
never silently shortened. Permanent input or permission failures are saved
and block another launch with unchanged context, even if the assistant
paraphrases the request or raises its effort. Other available capabilities may
still be used. If a repeated blocked action is attempted, the next response must
finish with the evidence and limitations already available.

The assistant and its tasks share one generated-token and active-time allowance.
The assistant's own calls run without thinking; delegated tasks use the effort
chosen for their work. Reaching a budget or context limit
retains partial work and its status. The standalone Prove notebook and terminal
proof baseline retain their existing solve, review and repair workflow.

The assistant's answers, direct or after a worker returns, have an output
ceiling of 8,192 tokens.
If the assistant's answer reaches its ceiling, Resume requests a complete
replacement using the saved results, without rerunning completed workers.
Its ceiling can grow to 16,384 tokens, within the remaining shared budget and
context space. The unfinished fragment is retained for inspection rather than
used as a proved premise.

For an OpenAI-compatible backend such as vLLM, `--assistant-concurrency` sets the
maximum number of simultaneous tasks, from 1 to 4 (default 2). Tasks that depend
on earlier results still wait for those results. Ollama and llama.cpp use a limit
of 1. All tasks use the same served model and reserve part of the parent budget;
concurrency does not multiply that budget or guarantee a speedup. Set the limit
to 1 to serialize inference, and measure the actual server before increasing it.

The shared limits are configurable at launch: `--assistant-tokens` (60,000
generated tokens), `--assistant-input-tokens` (240,000 input tokens), and
`--assistant-seconds` (900 seconds). A turn can create at most four workers and
use at most six tool actions. These limits apply across pause and resume.

Use **Pause**, or **Pause assistant** on the current task's card, to stop at a
checkpoint. **Resume assistant** continues the saved turn with its remaining
allowance, including a pending task. Resume that turn before sending another
message, or start a new conversation. A completed task's card stays finished
while the assistant continues. Older Default route cards remain readable and
retain their original start and cancel controls.

### Security and phones

By default the interface listens on `127.0.0.1` only. Every request needs the
workspace's access token, stored in `.mathagent/gui-token` (mode 0600) and
included in the printed address; the browser keeps it as an HttpOnly,
SameSite=Strict cookie and removes it from the address bar. Requests from other
web sites and unexpected `Host` headers are refused, and a strict
Content-Security-Policy stops rendered model or document text from loading
anything outside the harness. Fonts and KaTeX are bundled, so the interface itself
never contacts another server. Online search is a switch you set per conversation
and per report. Assistant chooses effort automatically; ordinary critique and
exploration chats also offer a Thinking switch. In the interface search is on by default, so
literature tools may send short search queries to arXiv, Semantic Scholar, zbMATH,
Crossref and OpenCitations; untick it to keep a conversation or a report offline. Write-ups and
proofs never search. Launching with an explicit `--offline` locks it off, which
keeps unaided benchmarks unaided. A document or a model answer can never turn it
on. Python stays a launch-time permission (`--allow-python`), and proof jobs have
no tools at all. Delete `.mathagent/gui-token` to unpair devices.

To use a phone on the same network:

```bash
square-harness --gui --gui-host 0.0.0.0 --workspace ~/research/my-paper
```

Open the printed phone address once; the phone then stays paired. On iPhone,
use Share, then Add to Home Screen; on Android, use the browser menu's Install
or Add to Home screen. A home-screen app keeps its own storage, so it may ask
for the token once more. **This traffic is plain HTTP on your local network:**
others on the network can read it, and anyone with the token controls the
harness, including approved file writes. Use it only on a trusted network, or
reach the harness through an SSH tunnel or a VPN such as WireGuard or Tailscale.
For a GPU server, `ssh -L 8765:127.0.0.1:8765 server` keeps the default loopback
mode and serves the interface to your laptop at `http://127.0.0.1:8765`.

### Building the interface

The built interface is committed in `mathagent/gui/static`, so installing the
package needs no Node.js. To change it, install Node.js 22 and work in
`frontend/`:

```bash
cd frontend
npm ci
npm run dev      # hot reload on :5173, using square-harness --gui on :8765 for /api
npm test
npm run build    # type-check, then rebuild mathagent/gui/static and its licence notices
```

CI rebuilds the interface and fails if the committed files differ from the sources.
