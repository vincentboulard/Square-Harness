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
and referee workflows are described in [research.md](research.md); they are not
part of the proof benchmark.

Manuscript writes require approval. Optional Python execution requires separate
approval and is not sandboxed. Saved jobs can contain unpublished text; keep
research workspaces private. The repository ignores `.mathagent/`, but an external
backup or manual file copy can still include it.
