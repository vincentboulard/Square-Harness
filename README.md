# Square Harness

**A local-first, open-source harness for mathematical research.**

Square Harness connects a language model running through [Ollama](https://ollama.com/)
to a mathematical workspace. It can read manuscripts, explore proof strategies,
keep a record of unfinished work, and prepare literature and referee-report drafts.

**Early experimental version (0.4).** The goal is a reusable research tool that
mathematicians can inspect, adapt, and run themselves. Outputs require human
checking: a model's critique or agreement is not a proof certificate.

## What it does

- **Explore and critique:** work with local LaTeX and text files through a terminal interface.
- **Continue proof work:** retain statements, arguments, objections, and next steps
  across bounded, resumable solver/reviewer rounds.
- **Investigate literature:** read cached or retrieved sources and draft reports
  with source locations and explicit limitations.
- **Write up notes:** turn rough notes, drafts or PDFs into LaTeX in your own macros
  and style, with every paragraph traced to the notes and the result compiled.
- **Stay local by default:** external retrieval is opt-in; manuscript edits require approval.

## Quick start

You need **Python 3.10+**, Linux or macOS, and a running Ollama server.
Native Windows is not supported yet. Model memory requirements depend on the
chosen model and context size.

```bash
git clone https://github.com/vincentboulard/Square-Harness.git
cd Square-Harness
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

ollama pull qwen3.8:27b
mkdir -p ~/research/square-workspace
square-harness --workspace ~/research/square-workspace
```

If Ollama is not already running, start `ollama serve` in another terminal.
The current default is `qwen3.8:27b`; select a different installed model with
`--model <exact-tag>`. It must support the harness's Ollama tool-calling interface;
compatibility and mathematical quality vary by model. Weights are downloaded
separately and are not included in this repository.

Try a self-contained question:

```text
/critic Is every bounded sequence in L²(0,1) strongly precompact? Justify your answer.
/prove Prove that every bounded sequence in H¹(0,1) has a subsequence converging strongly in L²(0,1).
```

For a manuscript, point `--workspace` at its directory, then ask:

```text
/critic Read manuscript.tex and identify the first unsupported implication.
```

### Visual interface

The same harness also runs as a local web app:

```bash
square-harness --gui --workspace ~/research/square-workspace
```

Your browser opens on `http://127.0.0.1:8765`. Each mode is a notebook in the left
rail, and the Free notebook suggests the right one for each request. Proofs,
critiques, reports and write-ups appear as the model writes them, with typeset
LaTeX, the claim ledger, budgets, citation checks and approval dialogs for file
writes. Drop files onto the window and switch folders from the sidebar. It uses the same engine and saved jobs as the terminal, including jobs
started there, and works in phone browsers too: see the
[usage guide](docs/usage.md#visual-interface) before exposing it to your network.

The `mathagent` command remains an alias. You can also run `python -m mathagent`
from the checkout without installing the optional terminal UI dependencies.

## Main commands

| Command | Purpose |
| --- | --- |
| `/prove <goal>` | Start a saved proof attempt with fixed budgets |
| `/critic <question>` / `/explore <question>` | Check an argument or explore approaches |
| `/review` | Review the last answer in a fresh model context |
| `/proofs` / `/ledger <id>` / `/resume <id>` | Inspect and resume saved proof work |
| `/literature <topic>` / `/referee <request>` | Start a saved research/report workflow |
| `/writeup <instructions>` | Write up pinned notes as LaTeX in your template style |
| `/help` | List all commands |
| `square-harness --gui` | Open the visual interface instead of the terminal prompt |

Proof and research jobs are saved under `.mathagent/` in your workspace.
Ctrl+C pauses a job; resuming preserves its remaining budget.
Launch with `--online` for external research. Proof search additionally requires
`--proof-literature` to use literature tools.

See the [usage guide](docs/usage.md) for manuscript inputs, reports, budgets,
optional PDF/Python tools, and all command-line settings.

## Limits and privacy

Solver and reviewers currently use the same model in separate contexts. They can
share mistakes; no formal proof checker is integrated. Automated tests check
software behavior, not mathematical reliability or comprehensive literature coverage.

Online search queries leave your machine. A remote model host receives prompts
and excerpts. Optional Python execution requires approval and **is not sandboxed**.
Saved jobs can contain manuscript text: keep research workspaces private.

## Contributing and citation

Created by **Vincent Boulard**, with **Louis Carillo**.
Small contributions, reproducible bug reports, and shareable mathematical failure
cases are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).
Maintainers decide which changes enter this repository.

Citation metadata is available in [CITATION.cff](CITATION.cff).
See [CHANGELOG.md](CHANGELOG.md) for the current release scope.

## License

[Apache License 2.0](LICENSE). Free to use, modify, and redistribute for scientific
and commercial purposes under its terms. The license covers the harness code,
documentation, and bundled skills; models, dependencies, and retrieved papers
retain their own licenses. See [NOTICE](NOTICE) for attribution.
