# Changelog

## Unreleased

- Review has four kinds on one page: a quick check of a proof (independent
  verifier passes through different lenses, each alleged error re-checked, a
  verdict with located issues and suggested repairs; the proof is never
  rewritten); a review of a manuscript (the whole paper read part by part for an
  overview and its typos and presentation problems, without proof checks); a
  detailed review that then checks the proofs one at a time, main results first,
  from a map of the manuscript's statements, proofs and dependencies, at twice
  the budget; and a precise explanation of a result and its proof. Independent
  calls run concurrently on vLLM. The Assistant and the terminal start them as
  `quick_review`, `referee`, `detailed_review` and `explain`.
- An independent research review that only announces what it will check no
  longer counts as a review: the report stays `partial`.

- Literature check: one or two precise references for a known result, each
  graded (read, cited, located, contradicted, not found) from the passages
  actually read, with papers that cite the work found through Crossref, the
  OpenCitations index and arXiv. The Assistant delegates it as a `check` worker;
  critic and explore answers can call `check_reference`; `/check` in the terminal.
- Literature reports are rebuilt as verified, themed reading lists: code sweeps
  Crossref, arXiv, zbMATH and Semantic Scholar and follows citations, and the
  model scopes, screens, organises and annotates only records a source returned.
  OpenAlex is no longer used.

- Saved conversations can be deleted from the Assistant sidebar after
  confirmation. Active and queued work is protected, other open clients update
  immediately, and saved proof/report jobs and attached files are retained.

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
- Assistant replaces Default: a main mathematical agent answers elementary
  questions directly or delegates precise tasks to the existing modes, then
  inspects their results and follows up. Independent workers can run concurrently
  on vLLM (two by default, configurable from one to four), within one shared
  token/time budget. Paused work retains child identities, results and accounting.
  Standalone proof mode and its direct-inference benchmark are unchanged.
- Assistant answers after delegated work have room for a full response. Resuming
  a capped answer reuses the saved results with a larger, context-fitted output
  ceiling, without launching the completed proof again.
- Assistant can read supplied public web pages and their links or paginated
  bibliographies directly, without starting a literature report. Entries without
  hyperlinks are preserved, and extraction coverage is explicit.
- Delegated research, review and write-up objectives are separate from their
  exact saved conversation context. Long earlier reports no longer count toward
  the objective length limit. Typed permanent failures persist across resume
  and prevent repeated worker launches with paraphrased requests or more effort.
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
- Interface notebooks are now Assistant, Prove, Literature, Review and Write-up.
  Critic and explore are no longer notebooks; Assistant can delegate to them, and
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
