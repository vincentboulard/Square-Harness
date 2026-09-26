# Square Harness

**A local-first, open-source harness for mathematical research.**

Square Harness connects a language model running through [Ollama](https://ollama.com/), [llama.cpp](https://github.com/ggml-org/llama.cpp), or [vLLM](https://vllm.ai/)
to a mathematical workspace. It can read manuscripts, explore proof strategies,
keep a record of unfinished work, and prepare literature and referee-report drafts.

**Early experimental version (0.4).** The goal is a reusable research tool that
mathematicians can inspect, adapt, and run themselves. Outputs require human
checking: a model's critique or agreement is not a proof certificate.

## What it does

- **Explore and critique:** work with local LaTeX and text files through a terminal interface.
- **Continue proof work:** retain statements, arguments, objections, and next steps
  across bounded, resumable solver/reviewer rounds.
- **Compare proof strategies:** run isolated parallel branches with a shared token
  ceiling, or opt into [cooperative subproblem work](docs/usage.md#cooperative-proof-work),
  and benchmark them against direct model sampling.
- **Investigate literature:** read cached or retrieved sources and draft reports
  with source locations and explicit limitations.
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
| `/help` | List all commands |

Proof and research jobs are saved under `.mathagent/` in your workspace.
Sequential proof jobs can resume after Ctrl+C with their remaining budget;
parallel portfolios and cooperative parents are one-shot runs.
Launch with `--online` for external research. Proof search additionally requires
`--proof-literature` to use literature tools.

See the [usage guide](docs/usage.md) for manuscript inputs, reports, budgets,
optional PDF/Python tools, and all command-line settings.
For GPU experiments, see the [single-A10 first pilot](docs/a10.md),
[two-V100S Q8 setup](docs/ovh.md), and [benchmark protocol](docs/benchmark.md). GPU performance must be measured on the target server.

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
