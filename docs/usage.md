# Usage

Proof mode is the default. It solves one problem, reviews the full candidate,
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
conversations.

| Mode | What the page shows |
| --- | --- |
| Free | A conversation where the model suggests the workflow for each message; you confirm before anything runs (see below) |
| Prove | The selected answer and its review status; each attempt with its written answer (and thinking), the verifier's verdict, explanation and located issues, and the exact model calls; live model output while a call runs; a side panel with budgets and every retained candidate |
| Critic, Explore | A conversation with typeset answers, collapsible thinking and tool calls, a thinking toggle, and a fresh-context review of the last answer |
| Literature, Referee | The report with clickable citations that open the exact passage read, the controller's citation checks, manuscript coverage, evidence, plan, notes and budgets |
| Write-up | The LaTeX document with source marks that open the note lines each paragraph came from, compile errors, the PDF, TODOs and template macros (see [LaTeX write-ups](research.md#latex-write-ups)) |

New proofs, reports and write-ups take an **effort**: Low, Medium, High, Extra
high or Brezis. Medium is the launch budget (`--proof-rounds`, `--proof-tokens`,
`--research-tokens` and so on); the other levels scale it by 0.35, 2.5, 6 and 20,
within the engine's limits (at most 100 attempts or rounds, and always room for one
full solve and its review). The summary under the slider states the exact limits,
and Advanced limits lets you edit them, including a proof's solve and review output
ceilings and its context. The Brezis effort, named after Haïm Brezis, can run for
hours; pause it whenever you like. Suggestions in Free mode carry the same control.

A proof started in the interface is the same job as `/prove`: the model sees the
statement and the pinned files only, with thinking on and no tools. The form warns
when the statement names a workspace file that is not pinned. Proof jobs made by
v0.4 stay listed with their report, sources and files, but cannot be resumed.

Every job page also lists its saved files: exact request payloads, streamed
answers and intermediate results. Pause stops model work at the next checkpoint,
like Ctrl+C in the terminal; Resume continues with the remaining saved budget.
The model serves one task at a time, because time budgets count wall-clock time:
pause a running job before starting another. Approval dialogs replace the
terminal's `[y/N]` prompt for file writes and, with `--allow-python`, for Python.

Critic and explore conversations are saved in `.mathagent/chats/` so that they
survive a reload. As in the terminal, a turn that is paused or fails never enters
the model's context; the interface keeps it visible and marks it as discarded.

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

### Free mode

The ∀ notebook accepts any request. For each message the model makes one short
structured call that suggests a workflow (proof, critique, exploration,
literature report, referee report or write-up) and rewrites the request so that
it stands on its own: proof and report jobs never see the conversation. A card
shows the suggestion; you can edit the request, change the workflow and remove
files before pressing Start, or dismiss it. When the message is ambiguous, the
card asks one question instead.

One conversation can hold several jobs: prove a statement, then ask for a
literature report, then referee a manuscript. Jobs appear on their cards with a
live status and open in their own notebook; later suggestions see the earlier
requests and their outcomes. Critiques and explorations are answered in the
conversation itself. Only files you attached or named are pinned, even if the
model suggests others. Reading a message is allowed while a job runs (its few
seconds count toward that job's time budget); a job started meanwhile waits in
the queue shown under the running task, where it can be cancelled.

### Security and phones

By default the interface listens on `127.0.0.1` only. Every request needs the
workspace's access token, stored in `.mathagent/gui-token` (mode 0600) and
included in the printed address; the browser keeps it as an HttpOnly,
SameSite=Strict cookie and removes it from the address bar. Requests from other
web sites and unexpected `Host` headers are refused, and a strict
Content-Security-Policy stops rendered model or document text from loading
anything outside the harness. Fonts and KaTeX are bundled, so the interface itself
never contacts another server. Online search is a switch you set per conversation
(next to Thinking) and per report; it is off by default, starting from `--online`
when that flag is given. Launching with an explicit `--offline` locks it off, which
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
