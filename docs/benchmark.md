# Benchmark: direct inference and proof mode

Version 0.5 compares two procedures on identical statement-only problems:

| Arm | Procedure | Default generated-token ceiling per problem |
| --- | --- | ---: |
| `raw-single` | One direct, thinking-enabled solve | 32,768 |
| `proof` | The same initial solve, then verification and bounded repair | 120,000 |

Ten problems with one replicate produce **20 jobs**. Both arms use the same model,
context, initial prompt, initial output cap, sampling settings and problem seed.
Proof mode adds no system prompt or tools to that initial solve. Matching seeds
is a reproduction aid, not a guarantee of identical GPU outputs.

The total budgets are deliberately unequal: the question is whether additional
verification and repair help. This is not an equal-compute comparison. Report
actual output tokens, input tokens, elapsed time and mathematical quality together.
The 10-problem maximum is 1,527,680 generated tokens at the defaults; early stopping
can use much less. There are no cooperative or portfolio arms in this protocol.

## Dataset

Keep statements outside the repository, and keep reference answers, hints,
source notes and grading rubrics outside every solver workspace. A manifest is:

```json
{
  "version": 1,
  "name": "Statement-only pilot",
  "problems": [
    {"id": "P1", "statement": "problems/P1.md"}
  ]
}
```

An optional `sha256` on each problem is verified when supplied. The runner always
freezes the exact statement hash. IDs and paths must be unique. Statement files
are `.md`, `.txt` or `.tex` under the manifest directory; escaping paths and
unexpected fields are rejected. Inspect the actual files yourself: a loader
cannot detect a hint hidden in otherwise valid statement text.

Both initial calls use this instruction:

> Give a complete mathematical proof. Standard results may be used if clearly stated and their hypotheses checked. Do not cite the requested assertion, or an equivalent theorem, as a black box. If you cannot finish, identify the precise unproved step.

No internet, literature, Python or workspace tools are supplied in either arm.
The chosen inference server receives the problem and model requests.

## Software smoke test

Create two synthetic toy statements in a new directory:

```bash
python -m mathagent.benchmark --init-smoke /srv/square/smoke-statements
```

Validate a deliberately small configuration without model calls or run-output
writes:

```bash
python -m mathagent.benchmark \
  --manifest /srv/square/smoke-statements/manifest.json \
  --output /srv/square/runs/smoke-001 \
  --backend openai --host http://127.0.0.1:18020 --model qwen3.8-27b \
  --arms raw-single proof --ctx 16384 --predict 4096 --verify-tokens 4096 \
  --tokens 16000 --rounds 2 --seconds 600 \
  --workers 1 --max-in-flight 1 --dry-run
```

After validating the serving endpoint, remove `--dry-run` for this bounded smoke
test only. Four jobs allow at most 40,960 output tokens. Toy success checks the
software path; it does not measure performance on research mathematics.
The model metadata defaults are explicitly unrecorded in a generic smoke run.

## Scientific run

Use the [HyperQwen A10 wrapper](hyperqwen-a10.md) for its pinned fast profile.
The retained [llama.cpp A10](a10.md) and [two-V100S](ovh.md) wrappers use their own
weights, 32k context and smaller per-call allowances. Treat them as separate
numerical and resource conditions; do not pool their results.

The generic runner accepts explicit settings:

```bash
python -m mathagent.benchmark \
  --manifest /srv/square/data/manifest.json \
  --output /srv/square/runs/proof-v05-001 \
  --backend openai --host http://127.0.0.1:18020 --model qwen3.8-27b \
  --model-revision '<immutable-weights-revision>' \
  --server-image '<image-digest>' --dtype '<actual-serving-precision>' \
  --arms raw-single proof --ctx 40960 --predict 32768 \
  --verify-tokens 16384 --tokens 120000 --rounds 3 \
  --seconds 1800 --request-timeout 600 --seed 20260927 \
  --workers 1 --max-in-flight 1 --dry-run
```

Replace the metadata placeholders with measured identities. Inspect the plan,
then remove `--dry-run` only when ready to run. A new output directory is required;
there are no silent overwrites or automatic retries of failed benchmark jobs.
Keep the model weights, server build, reasoning policy and harness source fixed.
Run long jobs in a server-side service or persistent terminal session.

`--predict` is the initial solve/repair ceiling **for both arms**. `--verify-tokens`
includes verifier thinking and its written response. `--tokens` is the proof-job
ceiling, including all reviews and repairs. `--rounds` bounds solver attempts.
`--raw-seconds` optionally gives direct inference a separate time guard; record
that difference. `--workers` and `--max-in-flight` control independent jobs, not
multiple agents working on one proof. Start with one job at a time.

Context is a separate input-plus-output limit. The whole candidate must fit into
verification; silently clipping it would invalidate the review. A token limit,
time limit, context failure or interrupted request is an operational outcome and
must not be reported as a mathematical refutation.

## Results and grading

Retain:

- `plan.json`: statements, configuration, seeds and source identities.
- `outputs.jsonl` and `summary.json`: job outcomes, costs and aggregate counts.
- Each job's `answer.md`, requests and response streams; proof jobs also retain
  the initial answer, every revision and verification in their saved workspace.
- The anonymized grading submissions, blank grading sheet and separate private
  mapping from submission ID to problem/arm.

The selected candidate is exported directly; no extra model call rewrites it.
Its status can be unreviewed or disputed. A model-approved answer is not a human
grade. Keep reviewers independent of arm labels when possible, and state any
limits to blinding or mathematical review. Do not score a model's confidence as
correctness.

Track both **correct answers damaged by intervention** and **incorrect answers
repaired**, as well as ordinary final-answer scores. Retained alternatives are
not automatically successful submissions; distinguish the selected answer from
an answer recovered retrospectively by a human.

To isolate verification effects during development, review the exact saved direct
answers before generating new samples. Matching a seed alone does not achieve
that isolation. Development on known problems is not a held-out test: freeze the
policy before assessing fresh problems and repeated runs.

Version 0.4 benchmark artifacts remain evidence about that earlier engine. Its
30-job runs, advisor outcomes and export rules are not results for v0.5.
