# HyperQwen on A10: v0.5 proof experiment

This profile keeps the pinned fast A10 serving condition and compares
**one direct solve** with **solve–verify–repair**. Ten statement-only problems
produce 20 jobs. It replaces the earlier 30-job sequential/advisor protocol;
those results do not validate the new proof engine.

## Settings

| Setting | Direct (`raw-single`) | Harness (`proof`) |
| --- | ---: | ---: |
| Context per request, input plus output | 40,960 | 40,960 |
| Initial solve ceiling | 32,768 | 32,768 |
| Verification ceiling per call | — | 16,384 |
| Total generated-token ceiling | 32,768 | 120,000 |
| Maximum solver attempts | 1 | 3 |
| Job time guard | 1,800 seconds | 1,800 seconds |

The two initial calls use the same instruction, statement, temperature 0.6,
top-p 0.95 and problem seed, with thinking enabled and no tools. Repairs use the
same solver cap. Verification has its own prompt and a fresh context. The proof
arm has additional compute; this is not an equal-total-budget comparison.

The exact image, weights revisions and launch environment are in
[`deployment/hyperqwen-a10.lock.json`](../deployment/hyperqwen-a10.lock.json).
The retained server uses W4A16 fast weights, DFlash2 with seven draft tokens,
a 4 GiB KV cache and up to two scheduler slots. The benchmark runs **one job at
a time**. Server slot count is not a claim of parallel proof reasoning or speedup.

## Thinking and context

The pinned OpenAI-compatible server supports `thinking_token_budget`. The wrapper
sets and records:

```bash
export SQUARE_OPENAI_THINKING_FRACTION=0.75
export SQUARE_OPENAI_STRUCTURED_THINKING_FRACTION=0.75
```

These reserve at most 75% of a response allowance for reasoning: up to 24,576
thinking tokens in a solve and 12,288 in a verification. The remainder is available
for the written response. Thinking and writing both count toward the total cap.
This is a server-enforced transition, not a second request or a correctness guarantee.
Verify support before using these settings with another serving build.

The 40k context applies separately to each request. A review needs room for the
original problem, whole written proof and its own output allowance. A repair
also includes the candidate and objections. Long input can therefore exhaust
context before the job exhausts its aggregate token budget; inspect that status
rather than treating it as a mathematical failure.

For interactive use against an already running matching server:

```bash
square-harness --workspace ~/research/my-paper \
  --backend openai --host http://127.0.0.1:18020 --model qwen3.8-27b \
  --ctx 40960 --proof-solve-tokens 32768 --proof-verify-tokens 16384 \
  --proof-tokens 120000 --proof-rounds 3 --proof-seconds 1800 \
  --request-timeout 1800 --proof-file statement.tex \
  --prompt "Prove the statement in statement.tex."
```

## Prepare and launch

This wrapper is for the existing pinned HyperQwen deployment, not a general
GPU provisioner. It requires its real launch record (image, container identity
and environment). Keep the weights, source revision and server unchanged after
checking them. First run the small two-arm [software smoke test](benchmark.md#software-smoke-test)
on the actual endpoint, including structured thinking verification. A dry run
alone does not validate GPU fit, throughput or the serving protocol.

Inspect the ten-problem configuration without inference:

```bash
python scripts/run-hyperqwen-benchmark.py \
  --manifest /srv/square/data/manifest.json \
  --output /srv/square/runs/proof-v05-001 \
  --launch-record /srv/square/server/launch.json \
  --dry-run
```

Remove `--dry-run` to run after the smoke test. The wrapper checks the live
container and model/context against the launch configuration. An optional
`--acceptance <record.json>` is validated against the current source, launch and
thinking policy; the old overnight record cannot certify changed v0.5 code.
There is no requirement to fabricate or manually mark an acceptance record passed.

Use a server-side service or persistent terminal so closing SSH does not stop the
run. Keep reference proofs and grading notes outside the deployed dataset.
The output directory must be new. Preserve the saved plan, source snapshot,
launch identity, complete streams, candidate history and grading exports.

## Duration and interpretation

The 20 jobs allow at most **1,527,680 generated tokens**. Runtime depends on actual
stopping, prompt processing, review length and repairs. The job time guards sum
to ten hours; that is a ceiling from the configured guards, not a runtime forecast.
No new GPU run or mathematical score is implied by preparing the v0.5 profile.
Measure a small live run before choosing an overnight window.

Use the [benchmark protocol](benchmark.md) for independent grading. Report useful
repairs, damaged correct answers, false objections, actual tokens and time.
A review that finds no issue is a model judgment, not a proof certificate.
