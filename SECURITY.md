# Security

## Reporting a vulnerability

Please do not put vulnerability details, exploit code, credentials, or private
research material in a public issue.

If this repository's **Security → Report a vulnerability** button is available,
use it to send a private report. Otherwise, open an
[issue](https://github.com/vincentboulard/Square-Harness/issues) saying only that
you need a private security reporting channel. A maintainer will arrange the
channel before you share details. The project has no guaranteed response time.

## Boundaries

- Ordinary file tools restrict access to the selected workspace and request
  approval for file writes. Choose a workspace containing only material you
  intend the model to read.
- Optional Python execution requires `--allow-python`, online mode, and approval
  for each call. It is **not sandboxed**: approved code runs with your account's
  permissions and may access files and the network. Read code before approving it.
- Offline mode uses a loopback Ollama endpoint and disables external retrieval.
  Online mode permits external research requests; search queries and requested
  URLs leave your computer. A remote Ollama endpoint receives model inputs.
- Saved jobs and transcripts can contain manuscript text and model output. Keep
  them private and redact them before sharing a bug report.

Model output and retrieved documents can be wrong or contain hostile
instructions. Human review remains necessary, including for mathematical claims.
