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

An optional **`cooperative`** arm uses advisor-planned subproblems, dependency-aware
workers and a newly assembled proof of the original theorem. It is a different
algorithm, not a renamed portfolio. Enable it explicitly with `--arms cooperative`
or add it to the three original arms. Ten problems with all four arms produce
forty jobs and a 2.4-million-token aggregate ceiling at the defaults. Existing
A10/V100 launchers and the generic runner's default arms remain unchanged.

The cooperative arm uses the same statement-only inputs, model, sampling
settings, total token ceiling and blinded human grading. It does not use the
portfolio selector or its `--selection-tokens` allocation. Its whole assembled
proof must pass the existing fresh critic and final audit to be marked
`candidate_complete`; model approval still does not determine the human score.
Unfinished assembled candidates are exported for partial-credit assessment;
without an assembled candidate, the submission is empty. See
[cooperative proof work](usage.md#cooperative-proof-work) for its bounded pipeline.

## Hardware profiles

The [two-V100S Q8 profile](ovh.md) runs three branches concurrently. The separate
[A10 Q4 pilot](a10.md) retains all three logical branches but executes one at a
time. Both keep the same problem statements, token allocation, selection and
human grading. Q4 and Q8 are distinct numerical conditions and must be reported
separately. The saved `parallel` arm names the portfolio algorithm; inspect
`branch_concurrency` and the saved schedule before making any speedup claim.

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
  --backend llamacpp --host http://127.0.0.1:8000 --model square-qwen \
  --arms sequential --ctx 32768 --tokens 6000 --selection-tokens 1536 \
  --predict 2048 --max-predict 2048 --rounds 2 --seconds 600 \
  --workers 1 --max-in-flight 1 \
  --dry-run
```

After the serving acceptance checks pass, remove `--dry-run` to run this bounded
development check: two sequential jobs, at most 12,000 generated tokens in total.
It exercises the proof workflow, not the three-method scientific comparison.
A tight cap may legitimately leave a toy unfinished; inspect its diagnostics
before choosing a separately recorded budget for another run. Do not use the
60,000-token scored profile as the default installation test. Test portfolios
separately after this check passes, using an explicit budget and a new output
directory. No smoke run automatically launches the scored benchmark.

The context preflight uses a conservative byte estimate, not the checkpoint's
exact tokenizer. The GPU smoke run must verify actual server context handling,
reasoning, structured outputs and tool calls before the mathematical experiment.
There is no implicit text truncation to make a full-proof review fit.

For the two-V100S experiment, use the wrapper after the server passes the
[OVH acceptance checks](ovh.md). It fixes the scientific settings, checks the
current server against the acceptance record, and embeds both records in the plan:

```bash
python scripts/run-v100-benchmark.py \
  --manifest "$HOME/square-data/pilot-10-v1/manifest.json" \
  --output "$HOME/square-runs/pilot-q8-001" \
  --launch-record "$HOME/square-runs/server/launch.json" \
  --acceptance "$HOME/square-runs/server/acceptance.json"
```

The V100 condition uses the same Q8_0 weights, template, 32K context, sampling
settings and server for every arm. Quantization changes the numerical model: do
not pool these results with BF16 runs. Weights are identified by their complete
GGUF SHA-256 digest, not merely a model alias. The V100 wrapper retains three
branches and the original token partition; no scientific statement is changed.

The generic `python -m mathagent.benchmark` runner remains available for other
servers with explicit `--model-revision`, `--server-image` and `--dtype` metadata.
It does not infer those identities from a model alias.
`unrecorded` metadata is permitted for software smoke tests and flagged in the
plan. See the [OVH runbook](ovh.md) for server configuration.

`--workers` controls independent problem/arm jobs. `--branches` controls direct
samples and same-problem proof branches. Preflight requires
`workers × active branch width <= max-in-flight`, before any model traffic.
Active width defaults to `--branches`; `--branch-concurrency 1` executes the
same logical attempts in separate waves of one. Waiting attempts have not yet
started and do not spend their branch time window. The available portfolio time
is divided equally between waves, with selection time reserved separately.
Seeds and token partitions depend on logical branch number, not scheduling. Thus two
jobs with two branches require `--workers 2 --branches 2 --max-in-flight 4`.
Fixed partitions bound requests without cross-process semaphore ownership.
The dedicated A10 wrapper uses `--branches 3 --branch-concurrency 1
--max-in-flight 1`; the V100 wrapper retains width three.
Start with the hardware profile defaults and increase concurrency only after measuring GPU memory
and aggregate throughput on the actual server.

For the cooperative arm, `--cooperative-concurrency` controls active subproblem
workers (1–3), independently of portfolio `--branches` and `--branch-concurrency`.
Preflight uses the **largest active width among selected arms** in
`workers × width <= max-in-flight`. Thus two jobs with cooperative concurrency
three require `--max-in-flight 6`, even if portfolio branches are serialized.
The width is an upper bound: dependencies can leave fewer tasks runnable.

To inspect a cooperative smoke configuration without any inference or writes:

```bash
python -m mathagent.benchmark \
  --manifest /srv/square/smoke-statements/manifest.json \
  --output /srv/square/runs/cooperative-smoke-001 \
  --backend llamacpp --host http://127.0.0.1:8000 --model square-qwen \
  --arms cooperative --cooperative-concurrency 3 --max-in-flight 3 \
  --ctx 32768 --predict 2048 --max-predict 4096 \
  --tokens 16000 --seconds 1800 --workers 1 --dry-run
```

Remove `--dry-run` only for a deliberately selected software smoke run after
server acceptance. This does not launch the ten-problem benchmark. A small
allowance can stop before assembly or review, which is an incomplete outcome.

## Budgets, selection and failures

With three branches, 6,144 tokens are reserved for selection and each branch
receives 17,952 tokens. A raw branch can use its whole allocation in one response;
a proof branch spends its allocation across solver, critics and ledger calls.
Complete untruncated candidates that pass the fresh critic go directly to the
whole-proof auditor, with deterministic recording afterward. Partial work still
uses the recorder loop. This applies equally to sequential and portfolio proof
branches; it does not change their token ceilings or selection rule. Record the
code revision when comparing results from before and after this workflow change.
Each full candidate gets a separate review, capped at 2,048 tokens. Reviewer
sampling temperature is zero. Solver temperature is 0.6 and top-p is 0.95 unless
changed. Initial branch seeds match across direct and parallel conditions;
subsequent harness calls get their own saved seed sequence. A saved seed is a
reproduction aid, not a guarantee of identical GPU outputs across server builds
or different batching schedules.

The cooperative budget requires at least 8,192 tokens and is frozen in
`plan.json` with strategy version 1. For ceiling B, initial planning receives
`min(2048, B//16)` and repair planning receives `min(1536, B//32)`. The remaining
tokens allocate 40% to initial workers, 30% to initial assembly, 10% to one repair
worker, and the integer balance to final assembly. Unused allocations are not
reassigned. At B = 60,000 these are 2,048, 1,536, 22,566, 16,924, 5,641 and 11,285
tokens respectively. An explicit direct route uses B minus the initial planning
allowance instead. The advisor may propose distinct tasks but cannot enlarge the
global budget or bypass the final whole-proof checks. Malformed plans stop as an
execution error without dispatching workers. Other worker or stage execution
errors remain recorded even if a later model approves an assembled candidate.
The saved parent state also records cumulative time checkpoints: 10%, 45%, 70%,
75%, 85% and 100% of the wall-time guard for planning, workers, first assembly,
repair planning, repair work and final assembly. Unused early time remains
available before later checkpoints; token allocations do not move. Report a
binding time guard separately from token exhaustion.

Selection ranks model reviews `complete`, `uncertain`, then `gap`; malformed
reviews can only produce an explicitly unreviewed fallback. Ties use the frozen
candidate order. Truncated candidates are not treated as full submissions.
Candidates whose complete text cannot be reviewed within the context window
are excluded with `needs_context`. No oracle/reference answer helps selection.
The serial arm exports its audited final candidate when available, otherwise a
predeclared last written candidate; full partial text is retained when no
complete draft exists. All original artifacts remain available for inspection.

Caps are maxima, not targets: the runner never burns unused tokens artificially.
Recorder-only protocol corrections are charged to those same caps. Exhausted
protocol recovery is an execution error, not a mathematical failure or success;
the candidate, raw responses and correction diagnostics remain available.
Measured completion tokens include thinking. Missing or interrupted usage is
charged conservatively at the request's reserved output allowance, reported
separately from measured tokens. Input tokens, request counts and wall times are
also recorded. Equal output-token ceilings are **not equal GPU compute**:
harness calls can have more input/prefill work. Compare actual consumption and
rented GPU-hours alongside mathematical quality.

`--seconds` is a wall-time guard, not an equal-time experimental allocation.
The V100 wrapper starts with 14,400 seconds per job, a 7,200-second request
timeout and a 1,800-second selection reserve. These are ceilings, not runtime
predictions. Calibrate them on synthetic inputs before scoring and keep them
fixed across arms. If a guard binds, report the trial as time-limited; the token
ceiling alone no longer describes the effective resource limit. The generic
runner exposes `--request-timeout` and `--selection-seconds`; portfolio review
reserves are capped at one quarter of the job time. These settings and all
actual times are saved.
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
- Cooperative jobs additionally retain their parent state/report, exact plan,
  stage allocations, dependency outcomes and individual worker/assembly ledgers.
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
