# Changelog

## 0.5.1 — 2026-09-27

Reliability fixes for the solve → verify → repair engine; no new model roles and
no extra reviews. One review still decides, and uncertainty still stops the job.

- Repairs and continuations fit their output allowance to the context (floor:
  `--proof-min-solve-tokens`), so they no longer fail when the input would not fit
  beside a full 32k allowance, including on the conservative byte estimate used
  without an exact tokenizer. The initial solve is unchanged and still identical
  to direct inference.
- Continuations keep the written text and a marked final excerpt of the saved
  notes; when nothing useful fits, the original problem is solved afresh.
- The verifier samples at the solver temperature by default
  (`--proof-verify-temperature`) instead of greedy decoding.
- Verifier responses are bounded (explanation 4,000 characters, at most five
  issues), so they fit the room left after thinking.
- The verifier receives the solver's task as quoted reference, not as its own
  instruction.
- Default request timeout 1,800 s; another attempt starts only if the measured
  first cycle fits in the remaining time. CLI notice for unpinned files.

## 0.5.0 — 2026-09-27

A simpler proof engine: solve the complete problem, verify the written proof,
and repair a concrete objection. This release defines a new experimental
protocol; mathematical gains have not yet been established by a live benchmark.

- A substantial initial solve with thinking enabled; the benchmark matches its
  initial prompt and sampling settings to the direct-inference baseline.
- One whole-proof verifier with explicit verdicts, explanations and located
  issues. Positive explanation text no longer counts as an objection.
- Targeted repair or rebuttal, followed by another full review. Protocol failures
  remain distinct from mathematical objections.
- Immutable initial answers and retained revisions. Partial later work cannot
  silently replace a complete response in the exported answer.
- Deterministic recording, persistent usage accounting and bounded resume.
- Removal of cooperative and portfolio proof modes, model recorder/planner,
  checkpoint summarization and proof-mode tool loops.
- Separate solve and verification ceilings; default job allowance 120,000
  generated tokens over at most three attempts, with a 40,960-token context.
- Two benchmark arms, `raw-single` and `proof`, with explicit additional compute
  for verification and repair. Retained hardware profiles are separate conditions.
- Shorter README and manuals, updated citation metadata and `--version`.

Compatibility: old proof artifacts remain inspectable, but cannot resume in the
new engine. Removed strategy and worker options are not silently mapped to new
behaviour. Keep original code and settings when reproducing older experiments.

## 0.4.0 — initial public release candidate

Early local-first mathematical research workflows: manuscript critique,
exploration, saved proof attempts, literature/referee drafts, Apache-2.0 license,
citation metadata and automated tests.

Subsequent v0.4 development explored proof portfolios, cooperative advisors,
OpenAI-compatible and llama.cpp transports, GPU deployment profiles and saved
benchmarks. These experiments motivated the v0.5 simplification; their original
results and source snapshots should retain their original version identity.

Tests validate software behaviour, not mathematical correctness. Interfaces and
saved-state formats may evolve while the project remains experimental.
