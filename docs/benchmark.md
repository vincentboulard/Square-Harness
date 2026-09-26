# Mathematical benchmark pilot

This runner compares a frozen model checkpoint on identical problem statements:

| Arm | Procedure | Total generated-token ceiling |
| --- | --- | --- |
| `raw-best` | Three independent direct answers, then independent reviews and fixed selection | 60,000 |
| `sequential` | One existing solver/critic/ledger proof search | 60,000 |
| `parallel` | Three isolated proof searches, then the same reviews and selection as `raw-best` | 60,000 |

The default is **one replicate**: ten problems produce thirty jobs. With the
default ceilings, at most 1.8 million generated tokens are authorized. Use
`--replicates 3` for a later repeated experiment, up to 5.4 million tokens.
Thinking, critiques, bookkeeping, selection, and interrupted requests all count
against the same job allowance. The final answer from the first direct attempt
is also exported as **raw@1**, requiring no extra model call. Its budget is
17,952 tokens with the defaults. `--arms raw-single` is an optional cheaper
single-answer reference, capped by `--predict` (8,192 by default); do not describe
that optional arm as an equal-budget comparison.

The portfolio is an initial experiment in independent search, not a claim that
parallel agents improve mathematical reliability. Branches do not share their
live ledgers or synthesize fragments into a new proof. Each branch has its own
process, workspace, seed and fixed token allowance. Independent reviews select
an **existing full candidate**, retaining the exact text. A model's approval is
never a mathematical score or proof certificate.

## Private dataset

Keep statements and results outside the source checkout. Keep reference answers,
source-page notes and grading rubrics in a separate directory that is never
copied into a solver workspace. A manifest contains only:

```json
{
  "version": 1,
  "name": "My statement-only pilot",
  "problems": [
    {
      "id": "P1",
      "statement": "problems/P1.md",
      "sha256": "the SHA-256 digest of the exact UTF-8 statement"
    }
  ]
}
```

`sha256` is optional when constructing a dataset and verified when provided.
The runner always computes and freezes every statement hash. IDs and paths must
be unique. Statement files must be `.md`, `.txt` or `.tex`, contained in the
manifest directory, with no parent traversal or symlinks escaping that directory.
Unexpected manifest fields, including answers and solutions, are rejected.
The loader cannot determine whether a statement file accidentally contains a
solution: inspect the statement-only dataset before running it.

Every arm receives this exact common instruction and the same statement:

> Give a complete mathematical proof. Standard results may be used if clearly stated and their hypotheses checked. Do not cite the requested assertion, or an equivalent theorem, as a black box. If you cannot finish, identify the precise unproved step.

Each job gets a fresh workspace containing only its statement. The proof
workflows also have their generic role instructions and local ledger tools.
Literature tools, internet retrieval, Python and manuscript writes are not
available to the benchmark proof workflows. The chosen inference endpoint still
receives prompts; use the local endpoint on your rented GPU host.

## Preflight and run

Create a software smoke dataset of two synthetic toy statements:

```bash
python -m mathagent.benchmark --init-smoke /srv/square/smoke-statements
```

First validate a complete configuration without any network call or output
writes:

```bash
python -m mathagent.benchmark \
  --manifest /srv/square/smoke-statements/manifest.json \
  --output /srv/square/runs/smoke-001 \
  --backend openai --host http://127.0.0.1:8000 --model square-qwen \
  --dry-run
```

The context preflight uses a conservative byte estimate, not the checkpoint's
exact tokenizer. The GPU smoke run must verify actual server context handling,
reasoning, structured outputs and tool calls before the mathematical experiment.
There is no implicit text truncation to make a full-proof review fit.

After the server passes its smoke checks, run the frozen private manifest:

```bash
python -m mathagent.benchmark \
  --manifest /srv/square/pilot/manifest.json \
  --output /srv/square/runs/pilot-001 \
  --backend openai --host http://127.0.0.1:8000 --model square-qwen \
  --model-revision EXACT_WEIGHTS_COMMIT --server-image EXACT_IMAGE_DIGEST \
  --dtype bfloat16 --ctx 32768 --tokens 60000 \
  --predict 8192 --max-predict 8192 \
  --branches 3 --selection-tokens 6144 --replicates 1 \
  --workers 1 --max-in-flight 3
```

The serving settings must match the metadata supplied here; the runner does not
infer checkpoint identity, precision or an image digest from a model alias.
`unrecorded` metadata is permitted for software smoke tests and flagged in the
plan. See the [OVH runbook](ovh.md) for server configuration.

`--workers` controls independent problem/arm jobs. `--branches` controls direct
samples and same-problem proof branches. Preflight requires
`workers × branch width <= max-in-flight`, before any model traffic. Thus two
jobs with two branches require `--workers 2 --branches 2 --max-in-flight 4`.
Fixed partitions bound requests without cross-process semaphore ownership.
Start with the defaults and increase concurrency only after measuring GPU memory
and aggregate throughput on the actual server.

## Budgets, selection and failures

With three branches, 6,144 tokens are reserved for selection and each branch
receives 17,952 tokens. A raw branch can use its whole allocation in one response;
a proof branch spends its allocation across solver, critics and ledger calls.
Each full candidate gets a separate review, capped at 2,048 tokens. Reviewer
sampling temperature is zero. Solver temperature is 0.6 and top-p is 0.95 unless
changed. Initial branch seeds match across direct and parallel conditions;
subsequent harness calls get their own saved seed sequence. A saved seed is a
reproduction aid, not a guarantee of identical GPU outputs across server builds
or different batching schedules.

Selection ranks model reviews `complete`, `uncertain`, then `gap`; malformed
reviews can only produce an explicitly unreviewed fallback. Ties use the frozen
candidate order. Truncated candidates are not treated as full submissions.
Candidates whose complete text cannot be reviewed within the context window
are excluded with `needs_context`. No oracle/reference answer helps selection.
The serial arm exports its audited final candidate when available, otherwise a
predeclared last written candidate; full partial text is retained when no
complete draft exists. All original artifacts remain available for inspection.

Caps are maxima, not targets: the runner never burns unused tokens artificially.
Measured completion tokens include thinking. Missing or interrupted usage is
charged conservatively at the request's reserved output allowance, reported
separately from measured tokens. Input tokens, request counts and wall times are
also recorded. Equal output-token ceilings are **not equal GPU compute**:
harness calls can have more input/prefill work. Compare actual consumption and
rented GPU-hours alongside mathematical quality.

`--seconds` is a wall-time guard, not an equal-time experimental allocation.
Raw and parallel portfolios reserve a tail for selection. Socket timeouts and
active requests can delay interruption; do not expect immediate cancellation
of an entire batch. Individual failures remain in the result set and do not
trigger automatic retries. Record them in the denominator, distinguishing
transport errors, truncation, context failures and incomplete mathematics.

Output directories are one-shot. Existing directories, even failed runs, are
refused before inference. This runner deliberately has no automatic batch
resume: inspect retained records and create a new explicitly named/configured
experiment if needed. Resuming an individual proof ledger is possible through
the ordinary CLI, but is additional compute outside the original frozen trial.

## Results and blind assessment

Each output contains:

- `plan.json`: configuration, dataset hashes, code hashes and Git revision/dirty
  state when a checkout is available.
- `outputs.jsonl` and per-job `result.json`: statuses, measured usage, unknown
  usage charged at reserved limits, timing, chosen artifact and exact answer hash.
- Per-job `answer.md`, direct request/stream journals and proof ledgers.
- `grading/`: anonymized statements/answers and a blank `scores.csv` worksheet.
- `grading-key.private.json`: arm/replicate mapping, kept outside the grading
  directory; model judgments remain in the private run records.
- `summary.json`: software execution/usage totals, with no inferred correctness.

Freeze the problem-specific rubrics before opening model outputs. The worksheet
uses 0 = no substantial correct solution, 1 = useful correct progress with a
gap, 2 = complete correct solution. Record the first unsupported step and
adjudicate disagreements. Report exact per-problem outcomes, false-completion
claims, costs and selected-answer accuracy. An oracle "at least one correct
candidate" score requires grading all candidates and must be labeled separately.
Ten problems are a pilot; repeated seeds do not increase the number of distinct
problems. Historical training contamination cannot be excluded merely by
isolating the benchmark files at inference time.
