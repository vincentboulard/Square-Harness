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
