# Changelog

## Unreleased

- Visual interface (`square-harness --gui`): a local web app for proof, critic,
  explore, literature and referee work, with typeset LaTeX, live model output,
  the claim ledger, citation checks, pausing and approval dialogs. The same code
  serves desktop and phone browsers; it listens on loopback by default and every
  request needs the workspace's access token.
- Critic and explore conversations started in the interface are saved in
  `.mathagent/chats/`.
- Free mode: the model suggests the workflow for each request and restates it
  self-contained; the user confirms before it runs. One conversation can mix
  proofs, critiques and reports; jobs started while the model is busy are queued.
- Write-up workflow (`--mode writeup`, `/writeup`, interface tab): LaTeX from
  notes, drafts or PDFs in the user's `.sty`/`.cls`/`.tex` template, with source
  comments checked against read lines, a sandboxed latexmk compile, one repair
  round and a fresh review. Output files never replace existing ones.
- The interface can open other folders below `--gui-root`, choose per folder which
  file types the model may read, and save dropped files into the workspace.
- `read_file` reads locally extracted PDF text; `.sty` and `.cls` are text files.
- Effort levels (Low to Brezis) replace budget fields in the interface; online
  search is a per-conversation and per-report switch, locked off by `--offline`.
- The interface uses the model server chosen with `--backend` (Ollama, vLLM or
  llama.cpp) and the sampling options; pausing works with every backend.
  Parallel portfolios and the cooperative strategy remain terminal-only.

- Opt-in cooperative proof strategy: an advisor proposes precise subproblems,
  bounded workers follow their dependencies, and an assembly job checks the
  original theorem. One targeted repair is allowed under a fixed total budget.
- Separate `cooperative` benchmark arm with frozen allocations and blinded
  grading; default arms and existing A10/V100 launchers remain unchanged.
- Complete candidates that pass the fresh critic skip recorder inference and
  proceed to final audit; partial work keeps the bounded recorder workflow.

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
