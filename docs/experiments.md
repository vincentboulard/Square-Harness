# Numerical experiments

Test a claim, a conjectured constant or an intuition numerically, and, when the numbers
point to one, obtain an explicit counterexample checked in exact arithmetic. Numerical
evidence is never reported as proof.

```bash
python -m pip install -e '.[experiments]'      # NumPy, SciPy, SymPy, mpmath, matplotlib
square-harness --workspace ~/research/my-paper --mode experiment \
  --prompt "Is the best constant in ‖u‖² ≤ C‖u′‖² on H¹₀(0,1) equal to 1/π²? Search for functions that come close."
```

In the terminal, `/experiment <claim>` starts one; `/researches`, `/research-report <id>`
and `/research-resume <id>` list, read and continue experiments like reports. In the visual
interface, use the **Experiment** notebook (≈).

## Workflow

1. **Protocol.** Before any code, the model fixes the hypothesis, the quantity computed,
   what would support or refute the claim, a validation case with an exactly known answer,
   how discretisation error is controlled and which family of inputs is explored.
2. **Code.** One script using the tested helpers in `mathagent.numerics` (imported as `sq`):
   finite-difference Laplacians (1-D Dirichlet, Neumann, periodic; 2-D rectangle), smallest
   eigenvalues, Crank–Nicolson heat and leapfrog wave solvers with optional sources (controls),
   observability Gramians, convergence studies, adversarial maximisation and standard plots.
3. **Run.** In a separate, resource-limited process (see *Isolation*), after your approval
   unless you chose automatic runs. A failing script, or one that never validated its method,
   goes back to the model with its error output, within the run budget.
4. **Interpretation.** The model reads the recorded numbers against the protocol.
5. **Certificate.** If the data point to an explicit witness, the model writes it with exact
   values; a fixed checker decides it in exact or interval arithmetic; a fresh model call
   judges whether it encodes the original statement.

The **controller**, not the model, sets the final status from `results.json`:

| Status | Meaning |
| --- | --- |
| `certified_counterexample` | The checker certified the witness and the review judged it faithful to the statement. |
| `evidence_against` | Validated, converged numerics contradict the claim. Not a rigorous refutation. |
| `consistent` | Validated, converged numerics agree on the explored family. Not a proof. |
| `inconclusive` | Unconverged, or no clear verdict. |
| `validation_failed` / `unvalidated` | The method failed, or never ran, its known-answer check: numbers unreliable. |
| `run_failed` | The code did not run within its runs. |
| `not_run` | Code written but not run (experiments off, or declined). |

A model verdict of "refutes" with a failed validation or an unconverged quantity is reported
as unreliable or inconclusive, whatever the model wrote.

## Counterexample certificates

A certificate is a small JSON file: exact witness values (`symbols`), explicit witness
functions (`functions`), the statement's hypotheses evaluated at the witness
(`assumptions`) and its conclusion (`claim`). The checker (`mathagent/certify.py`, which
needs only SymPy and mpmath) certifies a counterexample when every hypothesis holds and
the claim fails. Each relation is decided from the exact difference of its sides:

- exactly, when SymPy reduces it to a rational number (or to zero for an equality);
- otherwise by outward-rounded interval arithmetic at 64 to 4096 bits, which certifies a
  strict sign only when the enclosure excludes zero.

Decimal literals are refused, integrals and sums must have closed forms, and expressions
are restricted to arithmetic and a short list of functions. Each job keeps
`certificate/certificate.json` with a copy of the checker; anyone can rerun
`python verify_certificate.py certificate.json` there.

What the checker establishes is the arithmetic. Whether the certificate says the same thing
as your statement (same hypotheses, same conclusion, same objects) is a separate judgement,
made by a model review: read the certificate before relying on it.

## Isolation and permissions

`--experiments ask` (default) shows each script and its isolation before it runs;
`auto` runs without asking when network isolation is available and asks otherwise;
`off` writes the protocol and code without running anything. The interface has the same
choice per job.

The strongest available method is used and saved with every run:

| Method | Where | Effect |
| --- | --- | --- |
| `bwrap` | Linux with bubblewrap installed | No network, read-only system, home folder hidden except the Python installation, writes only in the run folder. |
| `seatbelt` | macOS (`sandbox-exec`) | No network, writes only in the run folder. |
| `netns` | Linux (`unshare -rn`) | No network. **Files are not protected**: read the code before approving. |
| `none` | Elsewhere | Resource limits only; the code runs with your permissions. |

Every method also applies a CPU-time and wall-clock limit (`--experiment-seconds`, default
120), a memory limit (`--experiment-memory`, 2048 MB, enforced on Linux), a 64 MB file-size
limit, one numerical thread, a clean environment without your variables, and a Python-level
network guard. Install bubblewrap (`apt install bubblewrap`) for the strongest isolation.

## Budgets

| Option | Default |
| --- | ---: |
| `--experiment-runs` | 4 runs, including fixes |
| `--experiment-tokens` | 30,000 generated tokens |
| `--experiment-time` | 1,800 seconds |
| `--experiment-seconds` | 120 seconds per run |
| `--experiment-memory` | 2,048 MB per run |

Input tokens use `--research-input-tokens`. Each run folder under
`.mathagent/research/<id>/runs/` keeps `experiment.py`, `results.json`, `stdout.txt`,
`stderr.txt` and the figures (PNG and SVG); rerun with `python experiment.py` there.

## With proof mode

Two opt-in flags connect experiments to proofs. Both use the experiment options above, have
their own token budgets (listed in the proof report) and count toward the proof's time.

- `--proof-refute-first`: before solving, an experiment tests the statement. A certified
  counterexample stops the proof with status `refuted`. Validated results
  (`consistent`, `evidence_against`) are shown to the verifier as fallible context. The
  initial solve is unchanged, so it still matches direct inference.
- `--proof-test-objections`: when the verifier raises a concrete objection, an experiment
  tests the disputed assertion before the repair, and the repair receives its result.

In the interface these are the *Test numerically first* and *Test objections numerically*
switches of the Prove form.

## Limits

An experiment explores a finite family of inputs with finite precision. "Consistent" says
nothing outside the explored family; floating-point agreement is not proof; a model can
write a wrong but plausible script, which is why validation on a known case is required and
why every script and output is kept for you to read.
