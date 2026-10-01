# Changelog

## Unreleased

- Numerical experiments (`--mode experiment`, `/experiment`, Experiment notebook): a protocol
  fixed before any code, a script using tested helpers (`mathagent/numerics.py`: Laplacians,
  spectra, heat and wave solvers, observability Gramians, convergence studies, adversarial
  search, plots), sandboxed runs (`mathagent/sandbox.py`: bubblewrap, macOS sandbox or network
  namespace when available, plus CPU, memory and file limits), bounded fixes, and a status set
  by the controller from `results.json` (validation and convergence required).
- Counterexample certificates (`mathagent/certify.py`): an explicit witness is checked in exact
  rational or outward-rounded interval arithmetic; a fresh review judges whether it encodes the
  statement. The checker is standalone and saved with each job.
- `--experiments off|ask|auto` and per-run limits; the interface asks with an experiment
  approval that states the isolation in force.
- Proof mode can run an experiment first (`--proof-refute-first`; a certified counterexample
  stops with status `refuted`) or test a verifier objection before the repair
  (`--proof-test-objections`). The initial solve is unchanged.
- Default mode can route a request to an experiment. New optional extra `experiments`.

- Visual interface (`square-harness --gui`): a local web app for proof, critic,
  explore, literature and referee work, with typeset LaTeX, live model output,
  proof candidates with their whole-proof reviews, citation checks, pausing and
  approval dialogs. The same code serves desktop and phone browsers; it listens
  on loopback by default and every request needs the workspace's access token.
- Proof pages follow the v0.5 engine: each attempt shows its written answer and
  the verifier's verdict and located issues; the selected answer and its review
  status stay visible. Legacy v1 proof jobs are listed and readable, not resumable.
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
- Effort levels (Low to Poincaré) replace budget fields in the interface; online
  search is a per-conversation and per-report switch, locked off by `--offline`.
- The interface uses the model server chosen with `--backend` (Ollama, vLLM or
  llama.cpp), the sampling options and the `--proof-*` limits; pausing works
  with every backend.
- Interface notebooks are now Default, Prove, Literature, Review and Write-up.
  Free mode is renamed Default: for each message the model picks one or more jobs,
  each with an effort level, and starts them at once in order, without asking for
  confirmation; each job's card says what started and can cancel it. Critic
  and explore are no longer notebooks; Default answers questions with them, and
  earlier critique and exploration conversations stay listed and readable there.
  Referee reports are called reviews in the interface (the engine mode is still
  `referee`).
- Effort levels have fixed budgets instead of multiples of the launch defaults:
  from Low (1 attempt, 1 minute, 30,000 tokens) through Medium (3 attempts) to
  Poincaré (10 attempts, 2 hours, 200,000 tokens). A proof's per-call ceilings are
  reduced when needed so one full solve and its review always fit the budget.
- Any job can be started while the model is busy: proof, report and write-up
  forms, suggestions and resume buttons add it to the queue instead of refusing.
- Online search is on by default in the interface for new conversations and
  reports; untick it per conversation or report. An explicit `--offline` launch
  still locks it off. Write-ups never search.
- The job list and a job page's overview (budget, candidates, checks) can slide
  away to a thin spine on wide screens; the choice is remembered per browser.
- The logo opens an About page describing the harness.

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
- Separate repair allowance (`--proof-repair-tokens`). The HyperQwen A10 profile now
  uses 80,000 tokens per job: a 32,768-token solve and 16,384-token review, then at
  most one repair of 14,000 tokens plus its review. `run-hyperqwen-benchmark.py --arms`.

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
