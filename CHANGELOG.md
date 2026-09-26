# Changelog

## Unreleased

- Separate single-A10 Q4 pilot with pinned weights/build, one 32K inference slot,
  and its own deployment and benchmark launchers; the two-V100S Q8 profile stays available.
- Bounded branch scheduling preserves three attempts and token partitions while
  executing one branch at a time, with dispatch-based time windows and recorded scheduling.
- One-slot serving acceptance is labeled sequential capacity rather than parallel speedup.
- Serving acceptance supports the pinned llama.cpp token-state array; A10 logs
  retain GPU placement and memory diagnostics for live validation.

- OpenAI-compatible local model transport for vLLM, with streamed reasoning,
  tool calls, structured reviews and explicit token usage.
- Isolated parallel proof portfolios with fixed aggregate token budgets and
  selection of an existing candidate after independent review.
- Statement-only benchmark manifests, direct-sampling and harness comparison,
  saved run metadata and anonymized submissions for mathematical grading.
- Pinned two-V100S deployment using CUDA 12 llama.cpp and one shared Q8_0
  model, with three 32K context slots and recorded model/image identities.
- Explicit llama.cpp transport with exact formatted-prompt context checks,
  streamed reasoning, tool history, structured reviews and token accounting.
- Concurrent-capacity acceptance checks, longer configurable time guards, and
  a launch wrapper that binds the benchmark to the verified serving record.
  GPU fit, throughput and mathematical gains still require the actual server.

## 0.4.0 — initial public release candidate

Early experimental version; prepared for publication, not a stability or
mathematical reliability guarantee.

- Terminal-based manuscript critique and mathematical exploration with Ollama.
- Persistent proof attempts with bounded budgets, saved arguments, objections,
  separate model reviews, and interruption recovery.
- Literature and referee-report workflows with cached sources and citation checks.
- Offline operation by default, approved edits, and optional approved Python.
- Apache-2.0 licensing, installation documentation, citation metadata, and CI.

Interfaces and saved-state formats may evolve. Research outputs need independent
mathematical checking. Live model performance is not measured by the test suite.
