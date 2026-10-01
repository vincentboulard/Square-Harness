"""Small terminal UI; optional Rich/prompt_toolkit, plain fallback without installs."""
import argparse
import copy
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse, quote

from .agent import Agent, AgentError, Ollama
from .backends import OpenAICompatible, create_client
from .prompts import MODES
from .proof import ProofRunner
from .ledger import ProofStore
from .tools import Workspace
from .literature import LiteratureTools
from .research import ResearchRunner
from .writeup import WriteupRunner
from .experiment import ExperimentRunner
from . import __version__

# Never let model/file content inject terminal escape sequences.
def clean(text):
    return re.sub(r'[\x00-\x08\x0b-\x1f\x7f-\x9f]', '', str(text))


class UI:
    def __init__(self, show_thinking=False):
        self.console = self.session = self.spinner = None
        self.show_thinking = show_thinking
        self.thinking_chars = 0
        self.text_started = False
        try:
            from rich.console import Console
            from prompt_toolkit import PromptSession
            self.console = Console(highlight=False)
            if sys.stdin.isatty():
                self.session = PromptSession()
        except ImportError:
            pass

    def say(self, text):
        if self.console:
            self.console.print(clean(text), markup=False)
        else:
            print(clean(text), flush=True)

    def ask(self, prompt):
        return self.session.prompt(prompt) if self.session else input(prompt)

    def stop(self):
        if self.spinner:
            self.spinner.stop()
            self.spinner = None

    def approve(self, preview):
        self.stop()
        self.say('\n' + preview)
        try:
            return self.ask('Approve this action? [y/N] ').strip().lower() == 'y'
        except (KeyboardInterrupt, EOFError):
            return False

    def emit(self, kind, value):
        if kind == 'start':
            self.thinking_chars, self.text_started = 0, False
            if self.console and self.console.is_terminal:
                self.spinner = self.console.status('Waiting for model… Ctrl+C cancels')
                self.spinner.start()
            else:
                self.say('Waiting for model…')
        elif kind == 'thinking':
            self.thinking_chars += len(value)
            if self.show_thinking:
                self.stop()
                if self.thinking_chars == len(value):
                    self.say('\n[model thinking]')
                print(clean(value), end='', flush=True)
            elif self.spinner:
                self.spinner.update(f'Thinking… {self.thinking_chars:,} characters generated · Ctrl+C cancels')
        elif kind == 'text':
            self.stop()
            if not self.text_started:
                self.say('\n[answer]')
                self.text_started = True
            print(clean(value), end='', flush=True)
        elif kind == 'end':
            self.stop()
            self.say('')
        else:
            self.stop()
            prefix = {'tool': '→ ', 'result': '  ', 'notice': 'Note: '}.get(kind, '')
            self.say(prefix + str(value))


HELP = '''Commands
  /prove [goal]                      Solve, verify, then repair if needed (default mode)
  /critic, /explore [query]            Ordinary conversational modes
  /literature [topic]                 Saved bibliographical research and Markdown report
  /referee [task or manuscript]       Saved manuscript review with literature checks
  /writeup [instructions]             LaTeX write-up of --research-file notes in the --template-file style
  /experiment [claim]                Numerical experiment: protocol, validated code, sandboxed run, verdict
  /researches                        List saved research/report jobs
  /research-report <id>               Read a saved Markdown report without inference
  /research-resume <id>               Resume a research job with its saved budgets
  /skills                            Show the built-in report skills
  /resume [proof-id]                  Resume a saved proof with its remaining budget
  /proof-report [proof-id]            Read its saved report (works offline; /ledger alias)
  /proofs                            List saved proof jobs (works offline)
  /review                            Fresh-context critique of the last answer/debrief
  /think on|off                      Toggle chat reasoning; proof work always uses thinking
  /context N                         Change context allocation (e.g. 16384)
  /clear                             Clear conversation history
  /status                            Model, settings and latest token statistics
  /save transcript.json              Export history/settings inside workspace
  /paste                             Multiline input; finish with a line containing /end
  /help, /quit
Ctrl+C pauses proof work at its saved checkpoint; /resume continues it.
In ordinary chat, the unfinished turn is discarded. Approved actions remain.
Python in chat is opt-in (--allow-python), always approved per call, and NOT sandboxed.
Experiments (--experiments off|ask|auto, default ask) run in a separate, resource-limited
process without network; numerical evidence is never reported as proof.
Manuscript writes require approval. Proof checkpoints save automatically in .mathagent/.
Proof status means model review, never formal verification.
Offline is the default: a loopback model server and cached/local sources only.
External research requires --online. Proof mode uses pinned text and no tools.
Reports save automatically in .mathagent/research/; --output exports Markdown.
'''


def parser():
    p = argparse.ArgumentParser(description='Square Harness: a small local mathematics agent')
    p.add_argument('--version', action='version', version='Square Harness ' + __version__)
    p.add_argument('--backend', choices=('ollama', 'openai', 'llamacpp'), default='ollama',
                   help='Model protocol: ollama, openai (vLLM), or llamacpp (with exact context checks)')
    p.add_argument('--model', default=None, help='Served model ID (default: qwen3.8:27b for Ollama; square-qwen for other servers)')
    p.add_argument('--workspace', type=Path, default=Path.cwd())
    p.add_argument('--host', default=None, help='Server URL (default: localhost:11434 for Ollama; localhost:8000 for other servers)')
    p.add_argument('--request-timeout', type=float, default=1800, help='Maximum seconds per model request')
    p.add_argument('--seed', type=int, help='Base sampling seed; saved proof jobs retain their original seed')
    p.add_argument('--temperature', type=float, default=0.6, help='Solver/chat sampling temperature')
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--ctx', type=int, default=40960)
    p.add_argument('--predict', type=int, default=4096,
                   help='Ordinary-chat output limit, including thinking; proof has separate limits')
    p.add_argument('--max-rounds', type=int, default=8, help='Tool rounds per ordinary chat query (not proof rounds)')
    p.add_argument('--proof-rounds', type=int, default=3, help='Maximum proof attempts, including the initial solve and revisions')
    p.add_argument('--proof-tokens', type=int, default=120000, help='Total generated-token ceiling, including all thinking and verification')
    p.add_argument('--proof-seconds', type=float, default=1800, help='Time budget in seconds per new proof')
    p.add_argument('--proof-solve-tokens', type=int, default=32768,
                   help='Output ceiling for each full solve or revision, including thinking')
    p.add_argument('--proof-verify-tokens', type=int, default=16384,
                   help='Output ceiling for each whole-proof review, including thinking')
    p.add_argument('--proof-repair-tokens', type=int, default=None,
                   help='Output ceiling for each repair/later attempt, including thinking (default: solve tokens)')
    p.add_argument('--proof-min-solve-tokens', type=int, default=None,
                   help='Smallest output allowance for a repair/continuation fitted to the context (default: min(16384, solve tokens))')
    p.add_argument('--proof-verify-temperature', type=float, default=None,
                   help='Verifier sampling temperature (default: --temperature)')
    p.add_argument('--proof-file', action='append', default=[], metavar='PATH',
                   help='Pin a complete theorem source relative to workspace; repeat for multiple files')
    network = p.add_mutually_exclusive_group()
    network.add_argument('--online', action='store_true', help='Allow external literature/search requests; queries leave this computer')
    network.add_argument('--offline', action='store_true', help='Use a loopback model server and local/cached sources only (default)')
    p.add_argument('--research-file', action='append', default=[], metavar='PATH', help='Pin a local manuscript/source for a report; repeat for multiple files')
    p.add_argument('--template-file', action='append', default=[], metavar='PATH',
                   help='Write-up template (.sty, .cls, .tex or .bib) whose macros and style must be used; repeatable')
    p.add_argument('--research-rounds', type=int, default=6, help='Maximum investigation rounds per new report')
    p.add_argument('--research-tokens', type=int, default=24000, help='Total generated-token budget per new report')
    p.add_argument('--research-input-tokens', type=int, default=100000, help='Cumulative input-token budget per new report')
    p.add_argument('--research-seconds', type=float, default=900, help='Total time budget per new report')
    p.add_argument('--research-requests', type=int, default=12, help='Maximum external HTTP requests per new research or assisted-proof job')
    p.add_argument('--research-chars', type=int, default=30000, help='Maximum literature-tool response characters per job')
    p.add_argument('--experiments', choices=('off', 'ask', 'auto'), default='ask',
                   help='Numerical experiments: off (write code only), ask before each run (default), or auto '
                        '(run without asking when network isolation is available)')
    p.add_argument('--experiment-seconds', type=int, default=120, help='CPU and wall-time limit per experiment run')
    p.add_argument('--experiment-memory', type=int, default=2048, help='Memory limit per experiment run, in MB')
    p.add_argument('--experiment-runs', type=int, default=4, help='Maximum runs per experiment, including fixes')
    p.add_argument('--experiment-tokens', type=int, default=30000, help='Generated-token budget per experiment')
    p.add_argument('--experiment-time', type=float, default=1800, help='Total time budget per experiment, in seconds')
    p.add_argument('--proof-refute-first', action='store_true',
                   help='Before a proof, run a numerical experiment; a certified counterexample stops the proof')
    p.add_argument('--proof-test-objections', action='store_true',
                   help='Test a verifier objection numerically before each repair; results go to the repair')
    p.add_argument('--output', metavar='PATH', help='Export the research/referee Markdown report inside the workspace')
    p.add_argument('--mode', choices=MODES, default='prove')
    p.add_argument('--no-think', action='store_true')
    p.add_argument('--show-thinking', action='store_true')
    p.add_argument('--allow-python', action='store_true', help='Offer unsandboxed Python with per-call approval (Linux)')
    p.add_argument('--gui', action='store_true', help='Serve the visual interface in your browser instead of the terminal prompt')
    p.add_argument('--gui-host', default='127.0.0.1',
                   help='Interface address (default: %(default)s); 0.0.0.0 lets phones on your network connect with the access token')
    p.add_argument('--gui-root', type=Path, metavar='PATH',
                   help='With --gui, folders at or below this one can be opened in the interface (default: the workspace)')
    p.add_argument('--gui-port', type=int, default=8765, help='Interface port; the next free port is used if taken (default: %(default)s)')
    p.add_argument('--no-browser', action='store_true', help='With --gui, do not open a browser automatically')
    one_shot = p.add_mutually_exclusive_group()
    one_shot.add_argument('--prompt', help='Run one query or command and exit')
    one_shot.add_argument('--resume', nargs='?', const='', metavar='PROOF_ID',
                          help='Resume a saved proof and exit (default: latest active proof)')
    one_shot.add_argument('--research-resume', metavar='RESEARCH_ID', help='Resume a saved literature/referee/write-up/experiment job and exit')
    return p


def select_proof(workspace_root, proof_id=''):
    """Select an explicit job or the latest resumable job, then latest job."""
    if proof_id:
        return ProofStore.load(workspace_root, proof_id)
    jobs = sorted(ProofStore.list(workspace_root), key=lambda job: job.get('updated_at', ''), reverse=True)
    if not jobs:
        raise ValueError('No saved proof jobs in this workspace. Start one with /prove <goal>.')
    active = [job for job in jobs if job.get('status') in {'ready', 'active', 'running', 'paused', 'interrupted', 'error', 'pending'}]
    return ProofStore.load(workspace_root, (active or jobs)[0]['id'])


def show_ledger(ui, store):
    report = store.directory / 'report.md'
    ui.say(f'Proof {store.state["id"]} · {store.state["status"]}')
    if report.exists():
        ui.say(report.read_text(encoding='utf-8'))
    else:
        ui.say(json.dumps(store.state, ensure_ascii=False, indent=2))
    ui.say(f'Proof directory: {store.directory}')


def main():
    p = parser()
    args = p.parse_args()
    args.host = args.host or ('http://localhost:11434' if args.backend == 'ollama' else 'http://localhost:8000')
    args.model = args.model or ('qwen3.8:27b' if args.backend == 'ollama' else 'square-qwen')
    if (not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2
            or not math.isfinite(args.top_p) or not 0 < args.top_p <= 1
            or args.seed is not None and not 0 <= args.seed < 2 ** 31):
        p.error('Use temperature 0–2, top-p in (0, 1], and seed in [0, 2**31)')
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        p.error('Request timeout must be finite and positive')
    if args.ctx < 2048 or not 0 < args.predict < args.ctx - 1024 or not 1 <= args.max_rounds <= 32:
        p.error('Use ctx >= 2048, 0 < predict < ctx - 1024, and 1 <= max-rounds <= 32')
    if (not 1 <= args.proof_rounds <= 100 or args.proof_tokens < 512
            or not math.isfinite(args.proof_seconds) or args.proof_seconds <= 0
            or args.proof_solve_tokens < 128 or args.proof_verify_tokens < 128):
        p.error('Use proof-rounds 1–100, proof-tokens >= 512, finite proof-seconds > 0, and solve/verify limits >= 128')
    if args.proof_repair_tokens is not None and not 128 <= args.proof_repair_tokens <= args.proof_solve_tokens:
        p.error('--proof-repair-tokens must be between 128 and --proof-solve-tokens')
    if args.proof_min_solve_tokens is not None and not 128 <= args.proof_min_solve_tokens <= (args.proof_repair_tokens or args.proof_solve_tokens):
        p.error('--proof-min-solve-tokens must be between 128 and the repair allowance')
    if args.proof_verify_temperature is not None and (not math.isfinite(args.proof_verify_temperature)
                                                      or not 0 <= args.proof_verify_temperature <= 2):
        p.error('--proof-verify-temperature must be between 0 and 2')
    if (not 1 <= args.research_rounds <= 50 or args.research_tokens < 1024
            or args.research_input_tokens < 2048 or not math.isfinite(args.research_seconds)
            or args.research_seconds <= 0 or not 0 <= args.research_requests <= 100
            or not 1000 <= args.research_chars <= 1_000_000):
        p.error('Use research-rounds 1–50, research-tokens >= 1024, research-input-tokens >= 2048, positive finite research-seconds, research-requests 0–100, and research-chars 1000–1000000')
    host = urlparse(args.host)
    if host.scheme not in {'http', 'https'} or not host.hostname or host.username is not None or host.query or host.fragment:
        p.error('--host must be an http(s) model URL without credentials, query or fragment')
    if not args.online:
        try:
            local_host = host.hostname == 'localhost' or ipaddress.ip_address(host.hostname).is_loopback
        except ValueError:
            local_host = False
        if not local_host:
            p.error('Offline mode requires a localhost/loopback model --host; use --online explicitly for a remote server')
        if args.allow_python:
            p.error('--allow-python requires --online: unrestricted Python can access the network')
    if (not 5 <= args.experiment_seconds <= 3600 or not 256 <= args.experiment_memory <= 65536
            or not 1 <= args.experiment_runs <= 20 or args.experiment_tokens < 4000
            or not math.isfinite(args.experiment_time) or args.experiment_time <= 0):
        p.error('Use experiment-seconds 5–3600, experiment-memory 256–65536 MB, experiment-runs 1–20, '
                'experiment-tokens >= 4000 and a positive finite experiment-time')
    if args.gui:
        if args.prompt is not None or args.resume is not None or args.research_resume is not None or args.output:
            p.error('--gui cannot be combined with --prompt, --resume, --research-resume or --output')
        if not 0 <= args.gui_port <= 65535:
            p.error('--gui-port must be between 0 and 65535')
        from .gui import serve
        try:
            return serve(args)
        except (OSError, ValueError) as e:
            print(f'Error: {e}', file=sys.stderr)
            return 2
    ui = UI(args.show_thinking)
    try:
        literature = LiteratureTools(args.workspace, online=args.online,
            max_requests=args.research_requests, max_chars=args.research_chars)
        workspace = Workspace(args.workspace, ui.approve, args.allow_python, literature=literature)
        client = (Ollama(args.host, timeout=args.request_timeout) if args.backend == 'ollama'
                  else OpenAICompatible(args.host, timeout=args.request_timeout) if args.backend == 'openai'
                  else create_client('llamacpp', args.host, timeout=args.request_timeout))
        agent = Agent(client, workspace, args.model, args.ctx, args.predict,
                      not args.no_think, args.mode, args.max_rounds,
                      seed=args.seed, temperature=args.temperature, top_p=args.top_p)
    except (AgentError, OSError, ValueError) as e:
        ui.say(str(e))
        return 2
    ui.say(f'╭─ SQUARE HARNESS · v{__version__}\n│ {agent.model} · {args.host}\n'
           f'│ Workspace: {workspace.root}\n│ Context {agent.ctx} · max output {agent.predict} · mode {agent.mode}\n'
           f'│ Research: {"online" if args.online else "offline (local/cache only)"} · proof: solve → verify → repair\n'
           '╰─ /help for commands · proof progress saved automatically · manuscript edits require approval')
    if args.allow_python:
        ui.say('Python enabled: each snippet requires approval and runs with your account permissions.')
    model_checked = False
    proof_running = False
    research_running = False
    last_proof_id = ''
    last_research_id = ''
    one_shot = args.prompt is not None or args.resume is not None or args.research_resume is not None

    def fresh_library():
        # A job restores its own counters and restrictions. Do not let an
        # offline resume or another job's exhausted budget alter later jobs.
        nonlocal literature
        literature = LiteratureTools(workspace.root, online=args.online,
            max_requests=args.research_requests, max_chars=args.research_chars)
        workspace.literature = literature

    def ensure_model():
        nonlocal model_checked
        if not model_checked:
            models = client.models()
            if agent.model not in models:
                hint = (f'Run: ollama pull {agent.model}' if args.backend == 'ollama'
                        else 'Start vLLM with this --served-model-name, or choose an available model ID.')
                raise AgentError(f'Model {agent.model!r} is not served. Available: {", ".join(models) or "none"}\n' + hint)
            model_checked = True

    def unpinned_files(query):
        pinned = {str(Path(f)) for f in args.proof_file}
        names = dict.fromkeys(re.findall(r'(?<![\w/])[\w./-]+\.(?:tex|md|txt)\b', query))
        found = []
        for name in names:
            try:
                if name not in pinned and workspace.path(name).is_file():
                    found.append(name)
            except (OSError, ValueError):
                continue
        return found

    def proof_result(result):
        nonlocal last_proof_id
        last_proof_id = result['id']
        ui.say(f'\nProof {result["id"]} · {result["status"]}')
        if result.get('answer'):
            ui.say(result['answer'])
            ui.say(f'Selected answer: {result["proof_path"]}')
        if result.get('report'):
            ui.say(result['report'])
        ui.say('A model review is not a mathematical certificate. See the report for unresolved issues.')
        ui.say(f'Report: {Path(result["directory"]) / "report.md"}')
        if result['status'] in {'ready', 'running', 'paused', 'interrupted', 'error'}:
            ui.say(f'Saved job: /resume {result["id"]} (uses the remaining original budget)')
        else:
            ui.say(f'Inspect: /proof-report {result["id"]}. A new /prove starts a separate job.')

    def research_result(result, export=True):
        nonlocal last_research_id
        last_research_id = result['id']
        ui.say(f'\nResearch {result["id"]} · {result["status"]}')
        ui.say(result['report'])
        ui.say(f'Markdown report: {Path(result["directory"]) / "report.md"}')
        if result['status'] in {'paused', 'error', 'running'}:
            ui.say(f'Resume: /research-resume {result["id"]}')
        if args.output and export:
            dest = workspace.path(args.output)
            if dest.suffix.lower() != '.md':
                raise ValueError('--output must be a workspace-relative .md file')
            if dest.exists() and not ui.approve(f'Overwrite report {dest}?'):
                ui.say('Export skipped; the saved report is still available above.')
                return
            dest = workspace.path(args.output)
            dest.parent.mkdir(parents=True, exist_ok=True)
            evidence_folder = os.path.relpath(Path(result['directory']) / 'artifacts', dest.parent)
            exported = result['report'].replace('](artifacts/', '](' + quote(evidence_folder, safe='/') + '/')
            dest.write_text(exported, encoding='utf-8')
            ui.say(f'Exported {dest}')

    def experiment_options():
        return dict(permission=args.experiments, run_seconds=args.experiment_seconds,
                    memory_mb=args.experiment_memory, source_files=args.research_file,
                    max_rounds=args.experiment_runs, max_tokens=args.experiment_tokens,
                    max_seconds=args.experiment_time)

    def proof_experiments():
        if not (args.proof_refute_first or args.proof_test_objections):
            return None
        options = experiment_options()
        options.pop('source_files')
        return dict(options, refute_first=args.proof_refute_first, test_objections=args.proof_test_objections)

    def run_query(query):
        nonlocal proof_running, last_proof_id, research_running, last_research_id
        fresh_library()
        if agent.mode == 'prove':
            if args.output:
                raise ValueError('--output is for literature/referee reports; proof reports save automatically')
            if max(args.proof_solve_tokens, args.proof_verify_tokens) + 256 >= agent.ctx:
                raise ValueError('Proof context must exceed each solve/verify output limit plus room for the input')
            if args.proof_tokens < args.proof_solve_tokens + args.proof_verify_tokens:
                raise ValueError('--proof-tokens must cover one full solve and verification; reduce their explicit limits or raise the total')
            unpinned = unpinned_files(query)
            if unpinned:
                ui.say('Note: proof mode has no file tools; the model sees only pinned text. Not pinned: '
                       + ', '.join(unpinned) + '. Relaunch with --proof-file to include them.')
            ensure_model()
            last_proof_id = ''
            proof_running = True
            result = ProofRunner(agent, ui.emit).start(query, max_rounds=args.proof_rounds,
                max_tokens=args.proof_tokens, max_seconds=args.proof_seconds,
                max_predict=args.proof_solve_tokens, verify_tokens=args.proof_verify_tokens,
                min_solve_tokens=args.proof_min_solve_tokens, repair_tokens=args.proof_repair_tokens,
                verify_temperature=args.proof_verify_temperature, source_files=args.proof_file,
                experiments=proof_experiments())
            proof_running = False
            proof_result(result)
            agent.history.extend([{'role': 'user', 'content': query},
                                  {'role': 'assistant', 'content': result.get('answer') or result.get('report', '')}])
        elif agent.mode in {'literature', 'referee'}:
            if args.output and workspace.path(args.output).suffix.lower() != '.md':
                raise ValueError('--output must be a workspace-relative .md file')
            ensure_model()
            research_running = True
            last_research_id = ''
            result = ResearchRunner(agent, ui.emit).start(query, kind=agent.mode,
                source_files=args.research_file, max_rounds=args.research_rounds,
                max_tokens=args.research_tokens, max_input_tokens=args.research_input_tokens,
                max_seconds=args.research_seconds, max_requests=args.research_requests,
                max_chars=args.research_chars)
            research_running = False
            research_result(result)
            agent.history.extend([{'role': 'user', 'content': query}, {'role': 'assistant', 'content': result.get('answer') or result.get('report', '')}])
        elif agent.mode == 'experiment':
            if args.output:
                raise ValueError('--output is not used by experiments; the report and run folders save automatically')
            ensure_model()
            research_running = True
            last_research_id = ''
            result = ExperimentRunner(agent, ui.emit).start(query, **experiment_options())
            research_running = False
            research_result(result, export=False)
        elif agent.mode == 'writeup':
            if args.output and workspace.path(args.output).suffix.lower() != '.tex':
                raise ValueError('For a write-up, --output names the new .tex file')
            ensure_model()
            research_running = True
            last_research_id = ''
            result = WriteupRunner(agent, ui.emit).start(query, source_files=args.research_file,
                template_files=args.template_file, output=args.output or '',
                max_rounds=args.research_rounds, max_tokens=args.research_tokens,
                max_input_tokens=args.research_input_tokens, max_seconds=args.research_seconds,
                max_requests=args.research_requests, max_chars=args.research_chars)
            research_running = False
            research_result(result, export=False)
        else:
            if args.output:
                raise ValueError('--output is for literature/referee reports')
            ensure_model()
            agent.run(query, ui.emit)

    while True:
        try:
            query = (('/resume ' + args.resume) if args.resume is not None else
                     ('/research-resume ' + args.research_resume) if args.research_resume is not None else
                     args.prompt if args.prompt is not None else ui.ask(f'\n{agent.mode}> ').strip())
            if not query:
                if one_shot:
                    return 0
                continue
            if query.startswith('/'):
                command, _, rest = query.partition(' ')
                rest = rest.strip()
                if command in ('/quit', '/exit'):
                    return 0
                if command == '/help':
                    ui.say(HELP)
                elif command == '/clear':
                    agent.history = []
                    ui.say('Conversation cleared.')
                elif command == '/skills':
                    for name in ('literature', 'referee', 'writeup', 'experiment'):
                        ui.say(f'{name}: {Path(__file__).parent / "skills" / name / "SKILL.md"}')
                elif command == '/researches':
                    jobs = ResearchRunner.list(workspace.root)
                    if not jobs:
                        ui.say('No saved literature/referee jobs in this workspace.')
                    for job in jobs:
                        ui.say(f'{job["id"]} · {job["status"]} · {job.get("kind", "research")} · {job.get("goal", "")[:160]}')
                elif command == '/research-report':
                    if not rest and not last_research_id:
                        raise ValueError('Supply a research job ID; use /researches to list jobs')
                    result = ResearchRunner.inspect(workspace.root, rest or last_research_id)
                    research_result(result, export=result.get('kind') not in ('writeup', 'experiment'))
                elif command == '/research-resume':
                    if not rest and not last_research_id:
                        raise ValueError('Supply a research job ID; use /researches to list jobs')
                    research_running = True
                    fresh_library()
                    job_id = rest or last_research_id
                    kind = ResearchRunner.inspect(workspace.root, job_id).get('kind')
                    if kind == 'experiment':
                        result = ExperimentRunner(agent, ui.emit).resume(job_id, permission=args.experiments)
                    else:
                        result = (WriteupRunner if kind == 'writeup' else ResearchRunner)(agent, ui.emit).resume(job_id)
                    research_running = False
                    model_checked = False
                    research_result(result, export=kind not in ('writeup', 'experiment'))
                elif command in {'/' + m for m in MODES}:
                    agent.mode = command[1:]
                    ui.say('Mode: ' + agent.mode)
                    if rest:
                        run_query(rest)
                elif command == '/proofs':
                    jobs = ProofStore.list(workspace.root)
                    if not jobs:
                        ui.say('No saved proof jobs in this workspace.')
                    for job in jobs:
                        goal = ' '.join(job['goal'].split())
                        ui.say(f'{job["id"]} · {job["status"]} · rounds {job.get("rounds_started", 0)} · {goal[:160]}')
                elif command in ('/proof-report', '/ledger'):
                    show_ledger(ui, select_proof(workspace.root, rest or last_proof_id))
                elif command == '/resume':
                    store = select_proof(workspace.root, rest or last_proof_id)
                    last_proof_id = store.state['id']
                    proof_running = True
                    fresh_library()
                    # The runner restores the saved model/settings; do not
                    # validate an unrelated CLI default model before resuming.
                    result = ProofRunner(agent, ui.emit).resume(last_proof_id)
                    proof_running = False
                    model_checked = False
                    proof_result(result)
                    agent.history.extend([{'role': 'user', 'content': store.state['goal']},
                                          {'role': 'assistant', 'content': result.get('answer') or result.get('report', '')}])
                elif command == '/think' and rest in ('on', 'off'):
                    agent.think = rest == 'on'
                    ui.say('Chat thinking: ' + rest + '. Proof mode always keeps thinking enabled.')
                elif command == '/context':
                    ctx = int(rest)
                    if ctx <= agent.predict + 1024:
                        raise ValueError('Context must exceed output budget + 1024')
                    agent.ctx = ctx
                    ui.say(f'Context set to {ctx}; applied on next request. Larger context needs more memory.')
                elif command == '/status':
                    ui.say(json.dumps({'model': agent.model, 'context': agent.ctx, 'predict': agent.predict,
                                       'backend': args.backend, 'seed': agent.seed,
                                       'temperature': agent.temperature, 'top_p': agent.top_p,
                                       'proof_solve_tokens': args.proof_solve_tokens,
                                       'proof_verify_tokens': args.proof_verify_tokens,
                                       'proof_tokens': args.proof_tokens,
                                       'think': agent.think, 'mode': agent.mode,
                                       'online': literature.online, 'literature_usage': literature.stats,
                                       'last_request': agent.last_stats}, indent=2))
                elif command == '/save':
                    dest = workspace.path(rest or 'mathagent-transcript.json')
                    if dest.exists() and not ui.approve(f'Overwrite {dest}?'):
                        if one_shot:
                            return 0
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(json.dumps({'version': 1, 'model': agent.model, 'mode': agent.mode,
                        'backend': args.backend, 'seed': agent.seed, 'temperature': agent.temperature, 'top_p': agent.top_p,
                        'context': agent.ctx, 'predict': agent.predict, 'think': agent.think,
                        'messages': agent.history}, ensure_ascii=False, indent=2), encoding='utf-8')
                    ui.say(f'Saved {dest}')
                elif command == '/paste':
                    ui.say('Enter multiline text. Type /end on its own line to submit.')
                    lines = []
                    while True:
                        line = ui.ask('… ')
                        if line == '/end':
                            break
                        lines.append(line)
                    if lines:
                        run_query('\n'.join(lines))
                elif command == '/review':
                    if not agent.history:
                        raise ValueError('Ask a question first')
                    last_user = next(m['content'] for m in reversed(agent.history) if m['role'] == 'user')
                    last_answer = agent.history[-1]['content']
                    reviewer = copy.copy(agent)
                    reviewer.mode, reviewer.history = 'critic', []
                    ensure_model()
                    ui.say('Fresh-context model critique; this is not formal verification.')
                    result = reviewer.run('Audit this proposed answer. Re-read relevant files if needed.\n'
                                          f'QUESTION:\n{last_user}\n\nANSWER:\n{last_answer}', ui.emit)
                    agent.history.extend([{'role': 'user', 'content': 'Critique the preceding answer.'},
                                          {'role': 'assistant', 'content': result}])
                    agent.last_stats = reviewer.last_stats
                else:
                    ui.say('Unknown command or invalid argument. Use /help.')
            else:
                run_query(query)
            if one_shot:
                return 0
        except KeyboardInterrupt:
            ui.stop()
            if proof_running:
                ui.say('\nProof interrupted. Completed checkpoints and the original budget are saved.')
                try:
                    store = select_proof(workspace.root, last_proof_id)
                    last_proof_id = store.state['id']
                    ui.say(f'Resume: /resume {last_proof_id}\nProof directory: {store.directory}')
                except (OSError, ValueError):
                    ui.say('Use /proofs to inspect saved jobs; initialization may have stopped before creating a checkpoint.')
                proof_running = False
            elif research_running:
                ui.say('\nResearch interrupted. Saved evidence, budgets and partial report remain in .mathagent/research/.')
                ui.say('Use /researches to find its ID, then /research-resume <id>.')
                research_running = False
            else:
                ui.say('\nCancelled. Unfinished chat turn discarded; previously approved actions remain.')
            if one_shot:
                return 130
        except EOFError:
            ui.stop()
            return 0
        except (AgentError, OSError, ValueError) as e:
            ui.stop()
            ui.say(f'Error: {e}')
            if proof_running:
                ui.say('Saved proof checkpoints remain available through /proofs, /proof-report and /resume.')
                proof_running = False
            if research_running:
                ui.say('Saved research evidence and partial reports remain available through /researches.')
                research_running = False
            if one_shot:
                return 1
