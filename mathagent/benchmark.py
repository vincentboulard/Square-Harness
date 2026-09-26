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
from .backends import create_client, TokenBudgetError
from .proof import ProofRunner
from .tools import Workspace
from urllib.parse import urlparse

COMMON_INSTRUCTION = (
    'Give a complete mathematical proof. Standard results may be used if clearly stated '
    'and their hypotheses checked. Do not cite the requested assertion, or an equivalent '
    'theorem, as a black box. If you cannot finish, identify the precise unproved step.'
)
ARMS = ('raw-single', 'raw-best', 'sequential', 'parallel')


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
    p.add_argument('--init-smoke', type=Path, help='Create a new external dataset containing two synthetic toy statements')
    p.add_argument('--backend', choices=('openai', 'ollama'), default='openai')
    p.add_argument('--host', default='http://127.0.0.1:8000')
    p.add_argument('--model', default='square-qwen')
    p.add_argument('--model-revision', default='unrecorded', help='Immutable weights revision; required for a scientific run record')
    p.add_argument('--server-image', default='unrecorded', help='Serving image digest/version for the run record')
    p.add_argument('--dtype', default='bfloat16')
    p.add_argument('--arms', nargs='+', choices=ARMS, default=['raw-best', 'sequential', 'parallel'])
    p.add_argument('--ctx', type=int, default=32768)
    p.add_argument('--predict', type=int, default=8192)
    p.add_argument('--max-predict', type=int, default=8192)
    p.add_argument('--tokens', type=int, default=60000)
    p.add_argument('--selection-tokens', type=int, default=6144)
    p.add_argument('--branches', type=int, default=3)
    p.add_argument('--replicates', type=int, default=1)
    p.add_argument('--seed', type=int, default=20260926)
    p.add_argument('--temperature', type=float, default=0.6)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--seconds', type=float, default=1800)
    p.add_argument('--rounds', type=int, default=10)
    p.add_argument('--workers', type=int, default=1, help='Independent problem/arm jobs in flight')
    p.add_argument('--max-in-flight', type=int, default=3, help='Global cap on simultaneous model requests, including branches')
    p.add_argument('--no-think', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Validate every input and show the frozen plan without network or writes')
    return p


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
    if not 1 <= args.branches <= 8 or not 1 <= args.replicates <= 100 or not 1 <= args.workers <= 32 or not 1 <= args.max_in_flight <= 32:
        raise ValueError('Use branches 1-8, replicates 1-100, workers/max-in-flight 1-32')
    if not args.arms or len(set(args.arms)) != len(args.arms):
        raise ValueError('Choose distinct nonempty arms')
    width = args.branches if any(arm in {'raw-best', 'parallel'} for arm in args.arms) else 1
    if args.workers * width > args.max_in_flight:
        raise ValueError('workers times branch width must not exceed max-in-flight; reduce workers or branches')
    if args.ctx < 2048 or args.predict < 128 or args.max_predict < args.predict or args.max_predict >= args.ctx - 1024:
        raise ValueError('Use ctx >= 2048 and 128 <= predict <= max-predict < ctx - 1024')
    if args.tokens < 1024 or args.selection_tokens < args.branches * 128 or args.tokens - args.selection_tokens < args.branches * 512:
        raise ValueError('Token budget must leave at least 512 tokens per branch and 128 review tokens per candidate')
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not 1 <= args.rounds <= 100:
        raise ValueError('Use positive finite seconds and rounds 1-100')
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1:
        raise ValueError('Invalid sampling parameters')
    # Dry run intentionally does not contact /models or a tokenizer server.
    # This conservative byte envelope is not a model-specific token count.
    raw_share = (args.tokens - args.selection_tokens) // args.branches
    caps = [args.max_predict]
    if 'raw-best' in args.arms:
        caps.append(raw_share)
    for problem in data['problems']:
        prompt_bytes = len((COMMON_INSTRUCTION + problem['text']).encode()) + 512
        if any(cap >= args.ctx - 512 or prompt_bytes > (args.ctx - cap - 512) * 3 for cap in caps):
            raise ValueError(f'Context/output envelope too small for {problem["id"]}; reduce output budgets or increase ctx')
    settings = {key: value for key, value in vars(args).items() if key not in {'manifest', 'output', 'init_smoke', 'dry_run'}}
    code = {}
    package = Path(__file__).resolve().parent
    for path in sorted(package.rglob('*.py')):
        code[str(path.relative_to(package))] = hashlib.sha256(path.read_bytes()).hexdigest()
    checkout = None
    if (package.parent / '.git').exists():
        try:
            sha = subprocess.run(['git', '-C', str(package.parent), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
            dirty = subprocess.run(['git', '-C', str(package.parent), 'status', '--porcelain'], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
            checkout = {'commit': sha, 'dirty': bool(dirty)}
        except (OSError, subprocess.SubprocessError):
            pass
    plan = {'version': 1, 'manifest': {k: v for k, v in data.items() if k != 'problems'},
            'problems': [{k: v for k, v in item.items() if k != 'text'} for item in data['problems']],
            'settings': settings, 'common_instruction': COMMON_INSTRUCTION, 'code_sha256': code, 'checkout': checkout,
            'provenance_warning': ('Checkpoint/server provenance is incomplete; suitable for software smoke tests only.'
                if 'unrecorded' in {args.model_revision, args.server_image} else None),
            'output': str(output), 'jobs': len(data['problems']) * args.replicates * len(args.arms),
            'generated_token_ceiling': len(data['problems']) * args.replicates * sum(
                args.predict if arm == 'raw-single' else args.tokens for arm in args.arms),
            'context_check': 'Conservative byte estimate only; exact server tokenization is verified in the GPU smoke run.',
            'raw_first': 'First raw-best candidate retained as raw@1; no additional inference.'}
    return data, plan


class GatedClient:
    """Optional gate for direct requests; preflight bounds every process partition."""
    def __init__(self, client, gate):
        self.client, self.request_gate = client, gate
        self.host, self.backend = client.host, client.backend

    @property
    def timeout(self):
        return self.client.timeout

    @timeout.setter
    def timeout(self, value):
        self.client.timeout = value

    def models(self):
        return self.client.models()

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
    payload = {'model': args.model, 'messages': [{'role': 'user', 'content': goal}],
               'think': not args.no_think, 'stream': True,
               'options': {'num_ctx': args.ctx, 'num_predict': cap, 'temperature': args.temperature,
                           'top_p': args.top_p, 'seed': seed}}
    _json(path / 'request.json', payload)
    call = {'reserved_tokens': cap, 'status': 'running', 'seed': seed, 'stats': {}}
    _json(path / 'call.json', call)  # reserve durably before dispatch
    text, done, calls = '', False, False
    started = time.monotonic()
    stream = None
    try:
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
        call['status'] = 'complete' if reason in {'stop', None} and not calls else 'truncated' if reason == 'length' else 'unexpected_completion'
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


def _sequential(agent, goal, path, args):
    result = ProofRunner(agent).start(goal, max_rounds=args.rounds, max_tokens=args.tokens,
        max_seconds=args.seconds, source_files=('statement.txt',), max_predict=args.max_predict,
        allow_literature=False)
    usage, states = _proof_usage(agent.workspace.root)
    answer, candidate, complete = '', None, False
    if states:
        directory, state = states[-1]
        if state.get('status') == 'candidate_complete' and state.get('final_audit') and state['final_audit'].get('candidate'):
            candidate = state['final_audit']['candidate']
        else:
            # Pending work is newer than committed rounds and old rejected audits.
            if state.get('pending'):
                candidate = state['pending'].get('draft')
            if not candidate:
                for round_record in reversed(state.get('rounds', [])):
                    candidate = round_record.get('candidate') or round_record.get('draft')
                    if candidate:
                        break
            if not candidate and state.get('final_audit'):
                candidate = state['final_audit'].get('candidate')
        if candidate:
            artifact = directory / 'artifacts' / candidate
            if not artifact.exists():
                artifact = directory / candidate
            value = json.loads(artifact.read_text(encoding='utf-8'))
            answer = value.get('text', '')
            complete = bool(value.get('complete'))
        if not answer:
            # Last solver stream retains full text when no draft checkpoint committed.
            for call in reversed(state['calls']):
                if call['role'] == 'solver':
                    stream = directory / 'artifacts' / call['stream']
                    if not stream.exists():
                        stream = directory / call['stream']
                    chunks = []
                    for line in stream.read_text(encoding='utf-8').splitlines():
                        try:
                            chunks.append((json.loads(line).get('message') or {}).get('content') or '')
                        except json.JSONDecodeError:
                            break
                    if any(chunks):
                        answer, candidate = ''.join(chunks), call['stream']
                        break
    answer_path = path / 'answer.md'
    answer_path.write_text(answer, encoding='utf-8')
    outcome = {**usage, 'status': result['status'], 'proof_status': result['status'], 'answer_path': str(answer_path),
               'selected_artifact': candidate, 'transport_complete': complete,
               'proof_directory': result['directory']}
    if states:
        outcome['stop_reason'] = state.get('stop_reason')
        if state['status'] in {'paused', 'interrupted', 'error', 'needs_recovery', 'budget_violation'}:
            outcome.update(status='error', error=state.get('stop_reason') or 'Proof execution did not complete')
    return outcome


def run_job(job, output, args, gate):
    from .portfolio import run_proof_portfolio, select_candidates, branch_seed
    problem, arm, replicate, seed = job
    key = f'{problem["id"]}--r{replicate + 1}--{arm}'
    path = output / 'jobs' / key
    path.mkdir(parents=True)
    workspace = path / 'workspace'
    workspace.mkdir()
    (workspace / 'statement.txt').write_text(problem['text'], encoding='utf-8')
    goal = COMMON_INSTRUCTION + '\n\nSTATEMENT:\n' + problem['text']
    client = GatedClient(create_client(args.backend, args.host, timeout=min(600, args.seconds)), gate)
    agent = Agent(client, Workspace(workspace), args.model, args.ctx, args.predict, not args.no_think,
                  seed=seed, temperature=args.temperature, top_p=args.top_p)
    record = {'id': key, 'problem_id': problem['id'], 'statement_sha256': problem['sha256'],
              'arm': arm, 'replicate': replicate + 1, 'seed': seed, 'budget': args.predict if arm == 'raw-single' else args.tokens,
              'status': 'running', 'answer_path': str(path / 'answer.md')}
    _json(path / 'result.json', record)
    started = time.monotonic()
    try:
        if arm == 'raw-single':
            candidate = _raw(client, goal, path / 'raw-0', args, args.predict, seed)
            record.update(_usage([candidate['call']]), status=candidate['call']['status'], answer_path=candidate['answer_path'])
            if candidate['call'].get('error'):
                record['error'] = candidate['call']['error']
        elif arm == 'raw-best':
            share = (args.tokens - args.selection_tokens) // args.branches
            # Each attempt receives a fixed partition; unused tokens are never duplicated.
            branch_seconds = args.seconds - min(300, args.seconds / 4)
            with ThreadPoolExecutor(max_workers=args.branches) as pool:
                futures = [pool.submit(_raw, GatedClient(create_client(args.backend, args.host, timeout=min(600, branch_seconds)), gate),
                    goal, path / f'raw-{i}', args, share, branch_seed(seed, i), branch_seconds) for i in range(args.branches)]
                candidates = [future.result() for future in futures]
            record['raw_at_1_path'] = candidates[0]['answer_path']
            direct = _usage([item['call'] for item in candidates])
            record.update(direct)
            if direct['tokens_charged'] > args.tokens - args.selection_tokens or any(c['call']['status'] == 'budget_violation' for c in candidates):
                raise AgentError('Server exceeded the reserved raw generation budget')
            record['failed_requests'] = sum(c['call']['status'] == 'error' for c in candidates)
            if record['failed_requests'] == len(candidates):
                raise AgentError('Every direct generation failed; inspect the per-request journals')
            remaining_seconds = args.seconds - (time.monotonic() - started)
            if remaining_seconds <= 0:
                raise AgentError('Raw portfolio elapsed-time budget exhausted before selection')
            selected = select_candidates(client, model=args.model, candidates=candidates, goal=goal,
                ctx=args.ctx, token_budget=args.selection_tokens, output_dir=path / 'selection',
                seed=seed, temperature=0, top_p=args.top_p, max_seconds=remaining_seconds)
            record.update(selected)
            for name in ('tokens_charged', 'measured_completion_tokens', 'prompt_tokens', 'reserved_unmeasured_tokens'):
                record[name] = direct[name] + selected.get(name, 0)
            record['request_count'] = direct['request_count'] + len(selected.get('calls', []))
            record['candidates'] = [{'id': c['id'], 'complete': c['complete'], 'answer_path': c['answer_path'], 'status': c['call']['status']} for c in candidates]
        elif arm == 'sequential':
            record.update(_sequential(agent, goal, path, args))
        else:
            result = run_proof_portfolio(agent, goal, output_dir=path / 'portfolio', workers=args.branches,
                max_tokens=args.tokens, max_seconds=args.seconds, max_rounds=args.rounds,
                max_predict=args.max_predict, source_files=('statement.txt',), seed=seed,
                selection_tokens=args.selection_tokens, request_gate=None, selector_goal=goal)
            record.update(result)
            record['request_count'] = sum(len(branch.get('calls', [])) for branch in result.get('branches', [])) + len(result.get('selection', {}).get('calls', []))
        execution_errors = []
        if arm == 'raw-best':
            execution_errors += [c['call'].get('error', 'Direct request failed') for c in candidates if c['call']['status'] == 'error']
            execution_errors += [str(c.get('status')) + ' selector request' for c in record.get('calls', []) if c.get('status') in {'interrupted', 'error'}]
        elif arm == 'parallel':
            execution_errors += [b.get('error') or b.get('proof_status') or 'Branch failed' for b in record.get('branches', [])
                                 if b.get('status') in {'failed', 'interrupted'} or b.get('proof_status') in {'paused', 'interrupted', 'error', 'needs_recovery', 'budget_violation'}]
            execution_errors += [str(c.get('status')) + ' selector request' for c in record.get('selection', {}).get('calls', []) if c.get('status') in {'interrupted', 'error'}]
        if record['status'] == 'budget_violation':
            execution_errors.append(record.get('error') or 'Server exceeded the reserved output allowance')
        if execution_errors:
            record.update(workflow_status=record['status'], status='error', execution_errors=execution_errors,
                          error='; '.join(execution_errors))
        answer = Path(record['answer_path']) if record.get('answer_path') else None
        if answer and answer.is_file():
            text = answer.read_text(encoding='utf-8')
            (path / 'answer.md').write_text(text, encoding='utf-8')
        else:
            (path / 'answer.md').write_text('', encoding='utf-8')
        record['answer_path'] = str(path / 'answer.md')
    except Exception as exc:
        # One failed item must not discard independent benchmark results.
        record.update(status='error', error=f'{type(exc).__name__}: {exc}')
        if arm == 'sequential':
            record.update(_proof_usage(workspace)[0])
        (path / 'answer.md').touch(exist_ok=True)
        record['answer_path'] = str(path / 'answer.md')
    record['id'] = key
    record['wall_seconds'] = time.monotonic() - started
    record['answer_sha256'] = hashlib.sha256((path / 'answer.md').read_bytes()).hexdigest()
    if record.get('raw_at_1_path'):
        record['raw_at_1_sha256'] = hashlib.sha256(Path(record['raw_at_1_path']).read_bytes()).hexdigest()
    _json(path / 'result.json', record)
    return record


def export_grading(output, records, problems, seed):
    grading = output / 'grading'
    grading.mkdir()
    statements = {p['id']: p['text'] for p in problems}
    records = sorted(records, key=lambda r: r['id'])
    items = [(record, record['answer_path'], record['arm']) for record in records]
    items.extend((r, r['raw_at_1_path'], 'raw@1') for r in records if r.get('raw_at_1_path'))
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
    (grading / 'README.md').write_text('Blind human assessment. Score 0: no substantial correct solution; 1: useful correct progress with a gap; 2: complete correct solution. State the first unproved step. Empty output is not a proof. Model verdicts are intentionally excluded. Agree on problem-specific rubrics before viewing outputs.\n', encoding='utf-8')
    _json(output / 'grading-key.private.json', key)


def execute(args, data, plan):
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    _json(output / 'plan.json', plan)
    client = create_client(args.backend, args.host, timeout=min(600, args.seconds))
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
    # Fixed request partitions avoid cross-process semaphore leaks if a worker
    # is killed. A job runs either N branches or one selector, never both;
    # preflight requires workers * N <= max_in_flight.
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
