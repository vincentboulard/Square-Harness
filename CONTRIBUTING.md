# Contributing

Square Harness is an early research project. Small fixes, clear failure cases,
documentation improvements, and focused pull requests are welcome.

## Development

Use Python 3.10 or later on macOS or Linux. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[research]'
python -m unittest discover -s tests -v
```

The automated tests use mocks and temporary workspaces; they do not require a
running model.

The visual interface's source lives in `frontend/` and is built into
`mathagent/gui/static`, which is committed so that installing needs no Node.js.
After changing it, run `npm ci`, `npm test` and `npm run build` in `frontend/`
(Node.js 22) and commit the rebuilt files with your source change. For model-dependent changes, describe the model tag, Ollama
version, settings, and manual check in your pull request.

## Issues and pull requests

- Explain the problem and expected behavior. Discuss substantial changes in an
  issue before implementing them.
- For a mathematical failure, provide a shareable statement, exact command and
  settings, relevant output, and the first false or unjustified inference.
- Add a regression test when fixing a reproducible code bug. Keep changes focused.
- Do not upload private manuscripts, credentials, or unredacted session logs.

Maintainers review contributions and decide what is merged. Public access lets
others propose changes or maintain their own forks; it does not grant write
access to this repository. Contributions are submitted under the project's
[Apache License 2.0](LICENSE).
