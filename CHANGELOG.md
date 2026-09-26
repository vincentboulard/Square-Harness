# Changelog

## Unreleased

- OpenAI-compatible local model transport for vLLM, with streamed reasoning,
  tool calls, structured reviews and explicit token usage.
- Isolated parallel proof portfolios with fixed aggregate token budgets and
  selection of an existing candidate after independent review.
- Statement-only benchmark manifests, direct-sampling and harness comparison,
  saved run metadata and anonymized submissions for mathematical grading.
- Pinned OVH H100 serving recipe and live transport checks. GPU memory fit,
  throughput and mathematical gains remain to be measured on the server.

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
