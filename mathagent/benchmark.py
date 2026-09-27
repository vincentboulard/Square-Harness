"""Private, bounded mathematical benchmark runner; no reference answers are loaded."""
import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import subprocess
import time

from .agent import Agent, AgentError
from .backends import create_client, TokenBudgetError, openai_thinking_budget_policy
from .proof import ProofRunner, ProofContext, check_context
from .proof_policy import COMMON_INSTRUCTION, initial_request
from .tools import Workspace
from urllib.parse import urlparse

ARMS = ('raw-single', 'proof')


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_manifest(filename):
    """Allowlisted schema deliberately has no answer, solution, or grading field."""
    filename = Path(filename).resolve(strict=True)
    value = json.loads(filename.read_text(encoding='utf-8'))
    if (not isinstance(value, dict) or set(value) - {'version', 'name', 'problems'}
            or type(value.get('version')) is not int or value['version'] != 1
            or not isinstance(value.get('problems'), list) or not value['problems']
            or len(value['problems']) > 1000):
        raise ValueError('Expected version 1 manifest with a nonempty problems list and no answer fields')
    if 'name' in value and not isinstance(value['name'], str):
        raise ValueError('Manifest name must be text')
    problems, ids, paths = [], set(), set()
    for item in value['problems']:
        if not isinstance(item, dict) or set(item) - {'id', 'statement', 'sha256'}:
            raise ValueError('Problem fields are id, statement, and optional sha256 only')
        identifier, relative = item.get('id'), item.get('statement')
        if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', identifier) or identifier in ids:
            raise ValueError('Problem IDs must be unique simple names, at most 80 characters')
        if not isinstance(relative, str) or not relative:
            raise ValueError('Each problem needs a relative statement path')
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or any(p.startswith('.') for p in path.parts):
            raise ValueError('Statement paths must be nonhidden and relative without parent traversal')
        path = (filename.parent / path).resolve(strict=True)
        if not path.is_relative_to(filename.parent) or path in paths or not path.is_file():
            raise ValueError('Statement paths must be unique files contained in the manifest directory')
        if path.suffix.lower() not in {'.md', '.txt', '.tex'} or path.stat().st_size > 240000:
            raise ValueError('Statements must be UTF-8 md/txt/tex files up to 240000 bytes')
        content = path.read_text(encoding='utf-8')
        if not content.strip() or '\x00' in content or len(content) > 59000:
            raise ValueError('Statement must be nonempty text, at most 59000 characters')
        digest = hashlib.sha256(content.encode()).hexdigest()
        if 'sha256' in item and item['sha256'] != digest:
            raise ValueError(f'Statement checksum mismatch: {identifier}')
        ids.add(identifier)
        paths.add(path)
        problems.append({'id': identifier, 'statement': relative, 'sha256': digest, 'text': content})
    return {'version': 1, 'name': value.get('name', filename.stem), 'path': str(filename),
            'sha256': hashlib.sha256(filename.read_bytes()).hexdigest(), 'problems': problems}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--init-smoke', type=Path, help='Create an external dataset with two synthetic toy statements')
    p.add_argument('--backend', choices=('openai', 'ollama', 'llamacpp'), default='openai')
    p.add_argument('--host', default='http://127.0.0.1:8000')
    p.add_argument('--model', default='square-qwen')
    p.add_argument('--model-revision', default='unrecorded', help='Immutable weights revision for provenance')
    p.add_argument('--server-image', default='unrecorded', help='Serving image digest/version for provenance')
    p.add_argument('--dtype', default='unrecorded', help='Recorded precision/quantization condition; not a server setting')
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--ctx', type=int, default=40960)
    p.add_argument('--predict', type=int, default=32768, help='Identical solver output ceiling in both arms')
    p.add_argument('--verify-tokens', type=int, default=16384, help='Output ceiling per proof verifier call, including thinking')
    p.add_argument('--min-solve-tokens', type=int, default=None,
                   help='Smallest context-fitted repair/continuation allowance (default: min(16384, predict))')
    p.add_argument('--verify-temperature', type=float, default=None, help='Verifier sampling temperature (default: --temperature)')
    p.add_argument('--tokens', type=int, default=120000, help='Total generated-token ceiling for every proof role combined')
    p.add_argument('--rounds', type=int, default=3, help='Maximum proof attempts, including the initial solve')
    p.add_argument('--replicates', type=int, default=1)
    p.add_argument('--seed', type=int, default=20260927)
    p.add_argument('--temperature', type=float, default=0.6)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--seconds', type=float, default=1800, help='Wall-time guard per job')
    p.add_argument('--raw-seconds', type=float, default=None, help='Optional different time guard for raw-single')
    p.add_argument('--request-timeout', type=float, default=1800, help='Per-request timeout, bounded by the job time remaining')
    p.add_argument('--workers', type=int, default=1, help='Independent problem/arm jobs in flight')
    p.add_argument('--max-in-flight', type=int, default=1, help='Maximum model requests in flight; must be at least workers')
    p.add_argument('--dry-run', action='store_true', help='Validate inputs and print the plan without network or writes')
    return p


def _source_files():
    package = Path(__file__).resolve().parent
    return {str(path.relative_to(package)): path.read_bytes() for path in sorted(package.rglob('*.py'))}


def preflight(args):
    if args.manifest is None or args.output is None:
        raise ValueError('--manifest and --output are required')
    host = urlparse(args.host)
    if host.scheme not in {'http', 'https'} or not host.hostname or host.username is not None or host.query or host.fragment:
        raise ValueError('Host must be an http(s) URL without credentials, query or fragment')
    data = load_manifest(args.manifest)
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Output already exists; refusing to rerun or overwrite it. Choose a new output directory.')
    if output.is_relative_to(Path(data['path']).parent):
        raise ValueError('Output must be outside the input dataset directory')
    if not 1 <= args.replicates <= 100 or not 1 <= args.workers <= args.max_in_flight <= 32:
        raise ValueError('Use replicates 1-100 and 1 <= workers <= max-in-flight <= 32')
    if not args.arms or len(set(args.arms)) != len(args.arms) or any(arm not in ARMS for arm in args.arms):
        raise ValueError('Choose distinct arms from raw-single and proof')
    if args.ctx < 2048 or not 128 <= args.predict < args.ctx - 1024 or not 128 <= args.verify_tokens < args.ctx - 1024:
        raise ValueError('Context requires ctx >= 2048 and 128 <= predict/verify-tokens < ctx - 1024')
    if 'proof' in args.arms and args.tokens < args.predict + args.verify_tokens:
        raise ValueError('Proof token budget must cover one full solver and verifier call')
    if not 1 <= args.rounds <= 100 or args.tokens < 1024:
        raise ValueError('Use rounds 1-100 and tokens >= 1024')
    for name in ('seconds', 'request_timeout', 'raw_seconds'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be finite and positive')
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2 or not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        raise ValueError('Invalid sampling parameters')
    if args.min_solve_tokens is None:
        args.min_solve_tokens = min(16384, args.predict)
    if not 128 <= args.min_solve_tokens <= args.predict:
        raise ValueError('min-solve-tokens must be between 128 and predict')
    if args.verify_temperature is None:
        args.verify_temperature = args.temperature
    if not math.isfinite(args.verify_temperature) or not 0 <= args.verify_temperature <= 2:
        raise ValueError('verify-temperature must be between 0 and 2')
    # No tokenizer/model calls during dry run. Full later inputs are checked
    # by the proof engine when the actual candidate is available.
    for problem in data['problems']:
        prompt_bytes = len(json.dumps(initial_request(problem['text']), ensure_ascii=False).encode())
        if prompt_bytes > (args.ctx - args.predict - 512) * 3:
            raise ValueError(f'Context/output envelope too small for {problem["id"]}; reduce predict or increase ctx')
    settings = {key: value for key, value in vars(args).items() if key not in {'manifest', 'output', 'init_smoke', 'dry_run'}}
    code = {name: hashlib.sha256(raw).hexdigest() for name, raw in _source_files().items()}
    package = Path(__file__).resolve().parent
    checkout = None
    if (package.parent / '.git').exists():
        try:
            sha = subprocess.run(['git', '-C', str(package.parent), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
            dirty = subprocess.run(['git', '-C', str(package.parent), 'status', '--porcelain'], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
            checkout = {'commit': sha, 'dirty': bool(dirty)}
        except (OSError, subprocess.SubprocessError):
            pass
    return data, {'version': 2, 'protocol': 'square-harness-0.5',
            'manifest': {k: v for k, v in data.items() if k != 'problems'},
            'problems': [{k: v for k, v in item.items() if k != 'text'} for item in data['problems']],
            'settings': settings, 'common_instruction': COMMON_INSTRUCTION, 'code_sha256': code, 'checkout': checkout,
            'provenance_warning': ('Checkpoint/server provenance is incomplete; suitable for software smoke tests only.'
                if 'unrecorded' in {args.model_revision, args.server_image, args.dtype} else None),
            'output': str(output), 'jobs': len(data['problems']) * args.replicates * len(args.arms),
            'generated_token_ceiling': len(data['problems']) * args.replicates * sum(
                args.predict if arm == 'raw-single' else args.tokens for arm in args.arms),
            'context_check': 'Offline byte estimate for the initial prompt; runtime checks full review/repair inputs without truncating them.',
            'initial_call_comparison': 'Both arms use identical initial messages, model, seed, thinking, sampling and output allowance. Matching seeds do not guarantee identical server samples.',
            'resource_comparison': 'Proof receives extra tokens for verification and repair; total budgets are not matched to raw-single.',
            'openai_thinking_budget_policy': openai_thinking_budget_policy() if args.backend == 'openai' else {}}


class GatedClient:
    """Optional gate for direct requests; preflight bounds every process partition."""
    def __init__(self, client, gate):
        self.client, self.request_gate = client, gate
        self.host, self.backend = client.host, client.backend

    @property
    def thinking_budget_policy(self):
        return getattr(self.client, 'thinking_budget_policy', {})

    @thinking_budget_policy.setter
    def thinking_budget_policy(self, value):
        self.client.thinking_budget_policy = value

    @property
    def timeout(self):
        return self.client.timeout

    @timeout.setter
    def timeout(self, value):
        self.client.timeout = value

    def models(self):
        return self.client.models()

    def count_input_tokens(self, payload):
        counter = getattr(self.client, 'count_input_tokens', None)
        return counter(payload) if callable(counter) else None

    def stream(self, payload):
        with self.request_gate if self.request_gate is not None else nullcontext():
            yield from self.client.stream(payload)


def _usage(calls):
    measured = prompts = reserved = charged = 0
    for call in calls:
        stats = call.get('stats') or {}
        n, p = stats.get('eval_count'), stats.get('prompt_eval_count')
        allowance = call.get('reserved_tokens', 0)
        if type(n) is int and n >= 0:
            measured += n
            charged += n
        else:
            reserved += allowance
            charged += allowance
        if type(p) is int and p >= 0:
            prompts += p
    return {'tokens_charged': charged, 'measured_completion_tokens': measured,
            'prompt_tokens': prompts, 'reserved_unmeasured_tokens': reserved, 'request_count': len(calls)}


def _raw(client, goal, path, args, cap, seed, max_seconds=None):
    path.mkdir()
    max_seconds = args.seconds if max_seconds is None else max_seconds
    payload = {'model': args.model, 'messages': initial_request(goal),
               'think': True, 'stream': True,
               'options': {'num_ctx': args.ctx, 'num_predict': cap, 'temperature': args.temperature,
                           'top_p': args.top_p, 'seed': seed}}
    _json(path / 'request.json', payload)
    call = {'reserved_tokens': cap, 'status': 'running', 'seed': seed, 'stats': {},
            'max_seconds': max_seconds}
    _json(path / 'call.json', call)  # reserve durably before dispatch
    text, done, calls = '', False, False
    started = time.monotonic()
    stream = None
    try:
        try:
            call['context_check'] = check_context(client, payload)
        except (ProofContext, AgentError, OSError, ValueError):
            call['reserved_tokens'] = 0  # No generation was dispatched.
            raise
        _json(path / 'call.json', call)
        remaining = max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            call['reserved_tokens'] = 0
            raise AgentError('Raw generation was not dispatched before the wall-time deadline')
        client.timeout = min(client.timeout, remaining)
        stream = client.stream(payload)
        with (path / 'stream.jsonl').open('w', encoding='utf-8') as log:
            for event in stream:
                log.write(json.dumps(event, ensure_ascii=False) + '\n')
                log.flush()
                message = event.get('message') or {}
                text += message.get('content') or ''
                calls = calls or bool(message.get('tool_calls'))
                if event.get('done'):
                    done = True
                    call['stats'] = {k: event[k] for k in ('eval_count', 'prompt_eval_count', 'done_reason') if k in event}
                if time.monotonic() - started > max_seconds:
                    raise AgentError('Raw generation exceeded its elapsed-time guard')
        if not done:
            raise AgentError('Stream ended without a completion event')
        reason = call['stats'].get('done_reason')
        call['status'] = ('complete' if reason == 'stop' and not calls and text.strip()
                          else 'empty_output' if reason == 'stop' and not calls
                          else 'truncated' if reason == 'length' else 'error')
        if call['status'] == 'error':
            call['error'] = 'Unexpected completion reason or tool call in a tool-free request'
        count = call['stats'].get('eval_count')
        if type(count) is int and count > cap:
            raise TokenBudgetError(call['stats'], cap)
    except TokenBudgetError as exc:
        call.update(status='budget_violation', error=str(exc), stats=exc.stats)
    except (AgentError, OSError, ValueError) as exc:
        call.update(status='error', error=str(exc))
    finally:
        if stream is not None and hasattr(stream, 'close'):
            stream.close()
        call['seconds'] = time.monotonic() - started
        _json(path / 'call.json', call)
        (path / 'answer.md').write_text(text, encoding='utf-8')
    return {'id': path.name, 'text': text, 'complete': call['status'] == 'complete' and bool(text.strip()),
            'answer_path': str(path / 'answer.md'), 'call': call}


def _proof_usage(workspace):
    calls, states = [], []
    for path in sorted((workspace / '.mathagent' / 'proofs').glob('*/state.json')):
        state = json.loads(path.read_text(encoding='utf-8'))
        calls.extend(state['calls'])
        states.append((path.parent, state))
    return _usage(calls), states


def _proof(agent, goal, path, args):
    # Export the engine-selected immutable candidate, never the newest draft.
    result = ProofRunner(agent).start(goal, max_rounds=args.rounds, max_tokens=args.tokens,
        max_seconds=args.seconds, source_files=(), max_predict=args.predict,
        verify_tokens=args.verify_tokens, min_solve_tokens=args.min_solve_tokens,
        verify_temperature=args.verify_temperature)
    usage, states = _proof_usage(agent.workspace.root)
    answer_path = path / 'answer.md'
    answer_path.write_text(result['answer'], encoding='utf-8')
    selected = result.get('selected_candidate') or {}
    outcome = {**usage, 'status': result['status'], 'proof_status': result['status'],
               'answer_path': str(answer_path), 'selected_artifact': selected.get('artifact'),
               'transport_complete': bool(selected.get('transport_complete')),
               'proof_directory': result.get('directory'), 'proof_path': result.get('proof_path'),
               'report_path': result.get('report_path'), 'initial_candidate': result.get('initial_candidate'),
               'selected_candidate': result.get('selected_candidate')}
    if states:
        state = states[-1][1]
        outcome['stop_reason'] = state.get('stop_reason')
        if state.get('status') in {'paused', 'interrupted', 'error', 'needs_recovery', 'budget_violation'}:
            outcome.update(status='error', error=state.get('stop_reason') or 'Proof execution failed; inspect its saved journal')
    return outcome


def run_job(job, output, args, gate=None):
    problem, arm, replicate, seed = job
    key = f'{problem["id"]}--r{replicate + 1}--{arm}'
    path = output / 'jobs' / key
    path.mkdir(parents=True)
    workspace = path / 'workspace'
    workspace.mkdir()
    # Evidence only: the statement is sent once, through the shared request builder.
    (workspace / 'statement.txt').write_text(problem['text'], encoding='utf-8')
    job_seconds = (args.raw_seconds or args.seconds) if arm == 'raw-single' else args.seconds
    client = GatedClient(create_client(args.backend, args.host, timeout=min(args.request_timeout, job_seconds)), gate)
    agent = Agent(client, Workspace(workspace), args.model, args.ctx, args.predict, True,
                  seed=seed, temperature=args.temperature, top_p=args.top_p)
    record = {'id': key, 'problem_id': problem['id'], 'statement_sha256': problem['sha256'],
              'arm': arm, 'replicate': replicate + 1, 'seed': seed,
              'budget': args.predict if arm == 'raw-single' else args.tokens,
              'max_seconds': job_seconds, 'status': 'running', 'answer_path': str(path / 'answer.md')}
    _json(path / 'result.json', record)
    started = time.monotonic()
    try:
        if arm == 'raw-single':
            candidate = _raw(client, problem['text'], path / 'raw-0', args, args.predict, seed, max_seconds=job_seconds)
            record.update(_usage([candidate['call']]), status=candidate['call']['status'],
                          answer_path=candidate['answer_path'], transport_complete=candidate['complete'])
            if candidate['call'].get('error'):
                record['error'] = candidate['call']['error']
            if record['status'] == 'budget_violation':
                record.update(workflow_status='budget_violation', status='error')
        elif arm == 'proof':
            record.update(_proof(agent, problem['text'], path, args))
        else:
            raise ValueError(f'Unknown benchmark arm: {arm}')
        answer = Path(record['answer_path'])
        text = answer.read_text(encoding='utf-8') if answer.is_file() else ''
        (path / 'answer.md').write_text(text, encoding='utf-8')
    except Exception as exc:
        # Keep independent trials and conservatively account every saved request.
        record.update(status='error', error=f'{type(exc).__name__}: {exc}')
        if arm == 'proof':
            record.update(_proof_usage(workspace)[0])
        (path / 'answer.md').touch(exist_ok=True)
    record['answer_path'] = str(path / 'answer.md')
    record['wall_seconds'] = time.monotonic() - started
    record['answer_sha256'] = hashlib.sha256((path / 'answer.md').read_bytes()).hexdigest()
    _json(path / 'result.json', record)
    return record


def export_grading(output, records, problems, seed):
    grading = output / 'grading'
    grading.mkdir()
    statements = {p['id']: p['text'] for p in problems}
    records = sorted(records, key=lambda r: r['id'])
    items = [(record, record['answer_path'], record['arm']) for record in records]
    random.Random(seed).shuffle(items)
    key, rows = {}, []
    for index, (record, source, arm) in enumerate(items, 1):
        blind = f'submission-{index:04d}'
        folder = grading / blind
        folder.mkdir()
        (folder / 'statement.md').write_text(COMMON_INSTRUCTION + '\n\n' + statements[record['problem_id']], encoding='utf-8')
        path = Path(source)
        (folder / 'answer.md').write_text(path.read_text(encoding='utf-8') if path.is_file() else '', encoding='utf-8')
        rows.append([blind, f'{blind}/statement.md', f'{blind}/answer.md', '', '', ''])
        key[blind] = {'job_id': record['id'], 'arm': arm, 'problem_id': record['problem_id'], 'replicate': record['replicate']}
    with (grading / 'scores.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['submission_id', 'statement', 'answer', 'score_0_1_2', 'first_unproved_step', 'notes'])
        writer.writerows(rows)
    (grading / 'README.md').write_text('Ungraded submissions with randomized IDs; keep grading-key.private.json away from graders. This layout alone does not establish that an assessment was blinded. Score 0: no substantial correct solution; 1: useful correct progress with a gap; 2: complete correct solution. State the first unproved step. Empty output is not a proof. Model verdicts are intentionally excluded. Agree on problem-specific rubrics before viewing outputs.\n', encoding='utf-8')
    _json(output / 'grading-key.private.json', key)


def execute(args, data, plan):
    policy = openai_thinking_budget_policy() if args.backend == 'openai' else {}
    if policy != plan.get('openai_thinking_budget_policy', {}):
        raise ValueError('Thinking budget policy changed after preflight')
    source = _source_files()
    if {name: hashlib.sha256(raw).hexdigest() for name, raw in source.items()} != plan['code_sha256']:
        raise ValueError('Source changed after preflight; create a fresh plan')
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    _json(output / 'plan.json', plan)
    for name, raw in source.items():
        destination = output / 'source' / 'mathagent' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    client = create_client(args.backend, args.host, timeout=min(args.request_timeout, args.seconds))
    try:
        if args.model not in client.models():
            raise ValueError(f'Model {args.model!r} not offered by the configured server')
    except (AgentError, OSError, ValueError) as exc:
        _json(output / 'startup-error.json', {'error': str(exc)})
        raise
    jobs = []
    for rep in range(args.replicates):
        for index, problem in enumerate(data['problems']):
            seed = (args.seed + rep * 100003 + index * 1009) % (2**31 - 16)
            jobs.extend((problem, arm, rep, seed) for arm in args.arms)
    random.Random(args.seed).shuffle(jobs)
    records = []
    # Each job makes at most one request at a time; workers bounds concurrency.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, job, output, args, None): job for job in jobs}
        with (output / 'outputs.jsonl').open('w', encoding='utf-8') as log:
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                log.write(json.dumps(record, ensure_ascii=False) + '\n')
                log.flush()
                print(f'{record["id"]}: {record["status"]}', flush=True)
    export_grading(output, records, data['problems'], args.seed + 1)
    _json(output / 'summary.json', {'completed_jobs': len(records), 'error_jobs': sum(r['status'] == 'error' for r in records),
        'tokens_charged': sum(r.get('tokens_charged', 0) for r in records),
        'measured_completion_tokens': sum(r.get('measured_completion_tokens', 0) for r in records),
        'prompt_tokens': sum(r.get('prompt_tokens', 0) for r in records),
        'request_count': sum(r.get('request_count', 0) for r in records),
        'reserved_unmeasured_tokens': sum(r.get('reserved_unmeasured_tokens', 0) for r in records),
        'note': 'No mathematical scores inferred. Complete the blinded human grading sheet.'})
    return records


def init_smoke(path):
    path.mkdir(parents=True, exist_ok=False)
    statements = [('toy_identity', 'Prove that (x+1)^2 - x^2 = 2*x+1 for every real number x.\n'),
                  ('toy_odd_sum', 'Prove that the sum of the first n positive odd integers is n^2 for every integer n >= 1.\n')]
    items = []
    for identifier, text in statements:
        filename = identifier + '.txt'
        (path / filename).write_text(text, encoding='utf-8')
        items.append({'id': identifier, 'statement': filename, 'sha256': hashlib.sha256(text.encode()).hexdigest()})
    _json(path / 'manifest.json', {'version': 1, 'name': 'Synthetic software smoke test, not a research benchmark', 'problems': items})
    return path / 'manifest.json'


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.init_smoke:
            if args.manifest or args.output:
                raise ValueError('--init-smoke cannot be combined with --manifest or --output')
            print(init_smoke(args.init_smoke))
            return 0
        data, plan = preflight(args)
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        records = execute(args, data, plan)
        return 1 if any(r['status'] == 'error' for r in records) else 0
    except (AgentError, OSError, ValueError) as exc:
        print(f'Benchmark error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
