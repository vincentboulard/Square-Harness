"""Portfolio isolation, reservations and selection under a real local server."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock
import warnings

from mathagent.agent import Agent, AgentError, Ollama
from mathagent.backends import TokenBudgetError
from mathagent.ledger import ProofStore
from mathagent.portfolio import branch_seed, branch_schedule, run_proof_portfolio, select_candidates, _branch_result, _stop_processes, _GatedClient
from mathagent.tools import Workspace


PROOF = 'Let x be an arbitrary real number. Equality is reflexive, hence x=x.'


def waiting_worker(directory):
    directory = Path(directory)
    try:
        (directory / 'ready').write_text('ready')
        while True:
            time.sleep(0.05)
    except KeyboardInterrupt:
        (directory / 'paused').write_text('paused')


def review(verdict='complete', **changes):
    value = {'verdict': verdict, 'explanation': 'Reflexivity applies to every real number.',
             'first_invalid_step': '' if verdict != 'gap' else 'The arbitrary quantifier is missing.'}
    value.update(changes)
    return value


def event(value, *, count=10, reason='stop', done=True):
    result = {'message': {'content': value if isinstance(value, str) else json.dumps(value)},
              'done': done, 'done_reason': reason, 'prompt_eval_count': 5}
    if count is not None:
        result['eval_count'] = count
    return result


class Client:
    timeout = 60

    def __init__(self, replies, on_request=None):
        self.replies, self.requests, self.on_request = iter(replies), [], on_request

    def stream(self, payload):
        self.requests.append(copy.deepcopy(payload))
        if self.on_request:
            self.on_request(payload)
        reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        yield from reply


class CandidateSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def select(self, client, candidates=None, **options):
        return select_candidates(client, model='test', goal='Prove x=x for all real x.',
            candidates=candidates or [{'id': 'a', 'text': PROOF, 'complete': True}],
            ctx=options.pop('ctx', 8192), token_budget=options.pop('token_budget', 1024),
            output_dir=options.pop('output_dir', self.root / 'selection'), **options)

    def test_blind_separate_reviews_exact_selection_and_reservation_before_dispatch(self):
        def check_reserved(payload):
            state = json.loads((self.root / 'selection' / 'state.json').read_text())
            self.assertEqual(state['calls'][-1]['status'], 'reserved')
            self.assertEqual(state['calls'][-1]['reserved_tokens'], 512)
            self.assertGreaterEqual(state['tokens_charged'], 512)
            self.assertEqual(payload['options']['temperature'], 0)
        client = Client([[event(review('gap'))], [event(review())]], check_reserved)
        candidates = [{'id': 'a', 'text': 'CANDIDATE_A. ' + PROOF, 'complete': True,
                       'prior_verdict': 'MARKER_FORBIDDEN'},
                      {'id': 'b', 'text': 'CANDIDATE_B. ' + PROOF, 'complete': True}]
        result = self.select(client, candidates)
        self.assertEqual(result['selected_id'], 'b')
        self.assertEqual(result['status'], 'selected_model_approved')
        self.assertEqual(Path(result['answer_path']).read_text(), candidates[1]['text'])
        self.assertNotIn('CANDIDATE_B', json.dumps(client.requests[0]))
        self.assertNotIn('CANDIDATE_A', json.dumps(client.requests[1]))
        self.assertNotIn('MARKER_FORBIDDEN', json.dumps(client.requests))
        self.assertEqual(result['tokens_charged'], 20)
        self.assertEqual(result['measured_completion_tokens'], 20)
        self.assertEqual(result['prompt_tokens'], 10)
        self.assertEqual(result['reserved_unmeasured_tokens'], 0)
        with self.assertRaises(FileExistsError):
            self.select(Client([]), candidates)

    def test_incomplete_candidates_never_dispatch_or_count_as_approved(self):
        client = Client([])
        result = self.select(client, [{'id': 'a', 'text': PROOF, 'complete': False}])
        self.assertEqual(result['status'], 'no_complete_candidate')
        self.assertIsNone(result['selected_id'])
        self.assertEqual(client.requests, [])

    def test_context_overflow_abstains_without_silent_clipping(self):
        client = Client([])
        result = self.select(client, [{'id': 'large', 'text': PROOF * 2000, 'complete': True}], ctx=1024)
        self.assertEqual(result['status'], 'needs_context')
        self.assertEqual(result['evaluations'][0]['verdict'], 'needs_context')
        self.assertIsNone(result['answer_path'])
        self.assertEqual(client.requests, [])

    def test_provider_context_rejection_also_abstains(self):
        result = self.select(Client([AgentError('maximum context length exceeded')]))
        self.assertEqual(result['status'], 'needs_context')
        self.assertEqual(result['tokens_charged'], 1024)
        self.assertIsNone(result['answer_path'])

    def test_llamacpp_exact_context_rejection_excludes_the_candidate(self):
        for index, message in enumerate(('Context budget exceeded: input plus output exceeds 32768',
                                         'Requested context 32768 exceeds llama.cpp per-slot context 8192')):
            with self.subTest(message=message):
                result = self.select(Client([AgentError(message)]), output_dir=self.root / str(index))
                self.assertEqual(result['status'], 'needs_context')
                self.assertIsNone(result['answer_path'])
                self.assertEqual(result['evaluations'][0]['verdict'], 'needs_context')

    def test_truncated_interrupted_and_contradictory_reviews_cannot_approve(self):
        for index, response in enumerate([
                [event(review(), reason='length')],
                [event(review(), done=False)],
                [event(review(first_invalid_step='An unjustified step exists.'))],
                [event('not JSON')]]):
            with self.subTest(index=index):
                result = self.select(Client([response]), output_dir=self.root / str(index))
                self.assertEqual(result['status'], 'selected_unverified')
                self.assertEqual(result['selected_status'], 'unreviewed')
                self.assertEqual(Path(result['answer_path']).read_text(), PROOF)
                if index == 1:
                    self.assertEqual(result['reserved_unmeasured_tokens'], 1024)

    def test_gap_candidate_remains_available_for_independent_grading(self):
        result = self.select(Client([[event(review('gap'), count=None)]]))
        self.assertEqual(result['selected_status'], 'gap')
        self.assertEqual(result['status'], 'selected_unverified')
        self.assertEqual(result['tokens_charged'], 1024)
        self.assertEqual(result['reserved_unmeasured_tokens'], 1024)
        self.assertEqual(result['measured_completion_tokens'], 0)

    def test_overreported_usage_cannot_produce_approval_or_further_calls(self):
        client = Client([[event(review(), count=700)]])
        result = self.select(client, [{'id': str(i), 'text': PROOF, 'complete': True} for i in range(2)])
        self.assertEqual(result['status'], 'budget_violation')
        self.assertIsNone(result['answer_path'])
        self.assertEqual(len(client.requests), 1)

    def test_adapter_budget_exception_retains_observed_usage_and_halts_selection(self):
        client = Client([TokenBudgetError({'eval_count': 700, 'prompt_eval_count': 17}, 512)])
        result = self.select(client, [{'id': str(i), 'text': PROOF, 'complete': True} for i in range(2)])
        self.assertEqual(result['status'], 'budget_violation')
        self.assertIsNone(result['answer_path'])
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(result['tokens_charged'], 700)
        self.assertEqual(result['measured_completion_tokens'], 700)
        self.assertEqual(result['prompt_tokens'], 17)
        self.assertEqual(result['reserved_unmeasured_tokens'], 0)

    def test_interrupt_keeps_reservation_and_does_not_replay(self):
        with self.assertRaises(KeyboardInterrupt):
            self.select(Client([KeyboardInterrupt()]))
        state = json.loads((self.root / 'selection' / 'state.json').read_text())
        self.assertEqual(state['status'], 'interrupted')
        self.assertEqual(state['tokens_charged'], 1024)
        with self.assertRaises(FileExistsError):
            self.select(Client([]))


class BranchInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / 'workspace'
        self.workspace.mkdir()
        self.job = {'id': 'branch-000', 'directory': str(self.root),
                    'workspace': str(self.workspace), 'seed': 1, 'token_budget': 1000}

    def store(self, *, complete=True, tokens=100):
        store = ProofStore.create(self.workspace, 'Prove x=x.',
                                  {'max_rounds': 1, 'max_tokens': 1000, 'max_seconds': 60})
        artifact = store.write_artifact('candidate', json.dumps({'text': PROOF, 'complete': complete, 'calls': []}))
        store.state['rounds'] = [{'index': 1, 'draft': artifact}]
        store.state['tokens_charged'] = tokens
        store.state['calls'] = [{'reserved_tokens': tokens, 'charged_tokens': tokens, 'stats': {}}]
        store.save()
        return store

    def test_missing_ambiguous_or_recovered_ledger_keeps_full_allocation(self):
        result = _branch_result(self.job)
        self.assertEqual(result['tokens_charged'], 1000)
        store = self.store()
        (store.directory / 'state.json').write_text('{broken')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = _branch_result(self.job)
        self.assertEqual(result['tokens_charged'], 1000)
        self.assertIsNone(result['candidate'])

    def test_provably_undispatched_branch_has_no_inference_charge(self):
        self.job['dispatched'] = False
        result = _branch_result(self.job)
        self.assertEqual(result['status'], 'not_dispatched')
        self.assertEqual(result['tokens_charged'], 0)
        self.assertIsNone(result['candidate'])

    def test_truncated_draft_is_ineligible(self):
        self.store(complete=False)
        self.assertIsNone(_branch_result(self.job)['candidate'])

    def test_recorder_protocol_failure_keeps_candidate_but_marks_branch_failed(self):
        store = self.store()
        store.state.update(status='stalled', stop_reason='Recorder repair failed.',
                           protocol_error={'stage': 'recorder', 'round': 1, 'attempts': 2})
        store.save()
        result = _branch_result(self.job)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['proof_status'], 'stalled')
        self.assertEqual(result['error'], 'Recorder repair failed.')
        self.assertEqual(result['protocol_error']['attempts'], 2)
        self.assertEqual(result['candidate']['text'], PROOF)
        self.assertEqual(result['tokens_charged'], 100)

    def test_overspend_is_not_selected_even_with_complete_draft(self):
        self.store(tokens=1001)
        result = _branch_result(self.job)
        self.assertTrue(result['budget_violation'])
        self.assertIsNone(result['candidate'])

    def test_request_cap_violation_below_branch_allocation_is_still_fatal(self):
        store = self.store(tokens=257)
        store.state['status'] = 'budget_violation'
        store.save()
        result = _branch_result(self.job)
        self.assertTrue(result['budget_violation'])
        self.assertEqual(result['tokens_charged'], 257)
        self.assertIsNone(result['candidate'])
        store.state['status'] = 'budget_exhausted'
        store.state['calls'][0]['status'] = 'budget_violation'
        store.save()
        self.assertTrue(_branch_result(self.job)['budget_violation'])

    def test_corrupt_worker_result_does_not_hide_durable_ledger_usage(self):
        self.store(tokens=123)
        (self.root / 'worker-result.json').write_text('broken')
        result = _branch_result(self.job)
        self.assertEqual(result['tokens_charged'], 123)
        self.assertIn('Worker result unavailable', result['error'])

    def test_rejected_old_audit_does_not_discard_newer_complete_draft(self):
        store = self.store()
        old = store.state['rounds'][0]['draft']
        newer_text = PROOF + ' This is the revised written candidate.'
        newer = store.write_artifact('candidate', json.dumps({'text': newer_text, 'complete': True, 'calls': []}))
        store.state.update(status='budget_exhausted', final_audit={
            'candidate': old, 'verdict': 'gap', 'objection': 'Earlier argument rejected.'})
        store.state['rounds'].append({'index': 2, 'draft': newer})
        store.save()
        result = _branch_result(self.job)
        self.assertEqual(result['candidate']['text'], newer_text)
        self.assertTrue(result['candidate']['artifact'].endswith(newer))
        pending_text = newer_text + ' A still newer pending revision.'
        pending = store.write_artifact('candidate', json.dumps({'text': pending_text, 'complete': True, 'calls': []}))
        store.state['pending'] = {'index': 3, 'phase': 'critic', 'draft': pending}
        store.save()
        self.assertEqual(_branch_result(self.job)['candidate']['text'], pending_text)


class PortfolioCancellationTests(unittest.TestCase):
    def test_selector_approval_cannot_hide_recorder_failure_or_override_budget_violation(self):
        for selection_status in ('selected_model_approved', 'budget_violation'):
            with self.subTest(selection_status=selection_status), tempfile.TemporaryDirectory() as root:
                agent = Agent(Ollama(), Workspace(root), predict=256)
                result_dir = Path(root) / 'portfolio'
                process = mock.Mock()
                process.is_alive.return_value = False
                context = mock.Mock()
                context.Process.return_value = process
                branch = {'id': 'branch-000', 'status': 'failed',
                          'protocol_error': {'stage': 'recorder', 'attempts': 2},
                          'error': 'Recorder repair failed.',
                          'candidate': {'id': 'branch-000', 'text': PROOF, 'complete': True},
                          'tokens_charged': 10, 'measured_completion_tokens': 10,
                          'prompt_tokens': 5, 'reserved_unmeasured_tokens': 0}

                def selector(*args, output_dir, candidates, **kwargs):
                    self.assertEqual(candidates, [branch['candidate']])
                    Path(output_dir).mkdir()
                    answer = Path(output_dir) / 'answer.md'
                    approved = selection_status == 'selected_model_approved'
                    if approved:
                        answer.write_text(PROOF)
                    return {'status': selection_status, 'selected_id': 'branch-000' if approved else None,
                            'selected_status': 'complete' if approved else None,
                            'answer_path': str(answer) if approved else None,
                            'tokens_charged': 13, 'measured_completion_tokens': 13,
                            'prompt_tokens': 7, 'reserved_unmeasured_tokens': 0}

                with mock.patch('mathagent.portfolio.multiprocessing.get_context', return_value=context), \
                        mock.patch('mathagent.portfolio._branch_result', return_value=branch), \
                        mock.patch('mathagent.portfolio.select_candidates', side_effect=selector):
                    result = run_proof_portfolio(agent, 'Prove x=x.', output_dir=result_dir,
                        workers=1, max_tokens=2000, selection_tokens=512, max_predict=512)
                self.assertEqual(result['workflow_errors'], ['branch-000: Recorder repair failed.'])
                self.assertEqual(result['status'], 'error' if selection_status == 'selected_model_approved'
                                 else 'budget_violation')
                if selection_status == 'selected_model_approved':
                    self.assertEqual(result['workflow_status'], selection_status)
                    self.assertIn('Recorder repair failed.', result['error'])
                    self.assertEqual(Path(result['answer_path']).read_text(), PROOF)
                else:
                    self.assertIsNone(result['answer_path'])
                self.assertEqual(result['selection']['status'], selection_status)
                self.assertEqual(result['tokens_charged'], 23)
                saved = json.loads((result_dir / 'state.json').read_text())
                self.assertEqual(saved['workflow_errors'], result['workflow_errors'])
                self.assertEqual(saved['status'], result['status'])

    def test_exhausted_request_gate_times_out_before_dispatch(self):
        client = Client([])
        gate = mock.Mock()
        gate.acquire.return_value = False
        with self.assertRaisesRegex(AgentError, 'request gate'):
            list(_GatedClient(client, gate).stream({'model': 'test'}))
        gate.acquire.assert_called_once_with(timeout=60)
        gate.release.assert_not_called()
        self.assertEqual(client.requests, [])

    def test_process_cleanup_interrupts_worker_and_drains_it(self):
        with tempfile.TemporaryDirectory() as root:
            process = multiprocessing.get_context('spawn').Process(target=waiting_worker, args=(root,))
            process.start()
            try:
                deadline = time.monotonic() + 5
                while not (Path(root) / 'ready').exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue((Path(root) / 'ready').exists())
                _stop_processes([process])
                self.assertFalse(process.is_alive())
                self.assertTrue((Path(root) / 'paused').exists())
            finally:
                _stop_processes([process])

    def test_parent_interrupt_during_selection_preserves_selection_reservation(self):
        with tempfile.TemporaryDirectory() as root:
            agent = Agent(Ollama(), Workspace(root), predict=256)
            result_dir = Path(root) / 'portfolio'
            process = mock.Mock()
            process.is_alive.return_value = False
            context = mock.Mock()
            context.Process.return_value = process
            branch = {'id': 'branch-000', 'candidate': {'id': 'branch-000', 'text': PROOF, 'complete': True},
                      'tokens_charged': 10, 'measured_completion_tokens': 10,
                      'prompt_tokens': 5, 'reserved_unmeasured_tokens': 0}

            def interrupted_selector(*args, output_dir, **kwargs):
                Path(output_dir).mkdir()
                (Path(output_dir) / 'state.json').write_text(json.dumps({
                    'status': 'interrupted', 'tokens_charged': 512, 'measured_completion_tokens': 0,
                    'prompt_tokens': 0, 'reserved_unmeasured_tokens': 512}))
                raise KeyboardInterrupt

            with mock.patch('mathagent.portfolio.multiprocessing.get_context', return_value=context), \
                    mock.patch('mathagent.portfolio._branch_result', return_value=branch), \
                    mock.patch('mathagent.portfolio.select_candidates', side_effect=interrupted_selector):
                with self.assertRaises(KeyboardInterrupt):
                    run_proof_portfolio(agent, 'Prove x=x.', output_dir=result_dir,
                        workers=1, max_tokens=2000, selection_tokens=512, max_predict=512)
            state = json.loads((result_dir / 'state.json').read_text())
            self.assertEqual(state['status'], 'interrupted')
            self.assertEqual(state['tokens_charged'], 522)
            self.assertEqual(state['reserved_unmeasured_tokens'], 512)


class ParallelProofIntegration(unittest.TestCase):
    def test_three_logical_branches_use_one_slot_and_start_only_when_dispatched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input'
            source.mkdir()
            result_dir = root / 'portfolio'
            lock = threading.Lock()
            requests, starts, queued_ledgers = [], [], []
            active = maximum = 0
            fail_seed = None
            overspend_seed = None

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_POST(self):
                    nonlocal active, maximum
                    payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    with lock:
                        active += 1
                        maximum = max(active, maximum)
                        requests.append(payload)
                    schema = payload.get('format')
                    if not schema:
                        state = json.loads((result_dir / 'state.json').read_text())
                        index = next(i for i, job in enumerate(state['jobs']) if job['seed'] == payload['options']['seed'])
                        starts.append((index, time.monotonic(), state['jobs'][index]['max_seconds']))
                        queued_ledgers.extend(list((Path(job['workspace']) / '.mathagent' / 'proofs').glob('*'))
                                              for job in state['jobs'][index + 1:])
                    # Long enough to distinguish actual sequential work from a
                    # parent that starts three children and queues HTTP calls.
                    time.sleep(0.12)
                    if not schema and payload['options']['seed'] == fail_seed:
                        self.send_response(503)
                        self.end_headers()
                        with lock:
                            active -= 1
                        self.wfile.write(b'Simulated first-branch failure')
                        return

                    def value(spec):
                        if spec['type'] == 'object':
                            return {key: value(child) for key, child in spec['properties'].items()}
                        if spec['type'] == 'array':
                            return []
                        if spec['type'] == 'boolean':
                            return False
                        if 'enum' in spec:
                            return 'gap' if 'gap' in spec['enum'] else spec['enum'][0]
                        return 'A mathematical step remains unproved.'

                    response = value(schema) if schema else PROOF
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/x-ndjson')
                    self.end_headers()
                    with lock:
                        active -= 1
                    count = payload['options']['num_predict'] + 1 if not schema and payload['options']['seed'] == overspend_seed else 10
                    self.wfile.write((json.dumps(event(response, count=count)) + '\n').encode())

            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                agent = Agent(Ollama(f'http://127.0.0.1:{server.server_port}'), Workspace(source),
                              model='test', ctx=32768, predict=256)
                result = run_proof_portfolio(agent, 'Prove x=x for every real x.', output_dir=result_dir,
                    workers=3, branch_concurrency=1, max_tokens=60000, selection_tokens=6144,
                    max_seconds=15, selection_seconds=3, max_rounds=1, max_predict=512, seed=7)
                self.assertEqual(maximum, 1)
                self.assertEqual([index for index, _, _ in starts], [0, 1, 2])
                self.assertTrue(all(seconds == 4 for _, _, seconds in starts))
                self.assertTrue(all(not files for files in queued_ledgers))
                self.assertEqual([job['seed'] for job in result['jobs']], [branch_seed(7, i) for i in range(3)])
                self.assertEqual([job['token_budget'] for job in result['jobs']], [17952] * 3)
                self.assertEqual([job['wave'] for job in result['jobs']], [0, 1, 2])
                self.assertTrue(all(job['dispatched'] for job in result['jobs']))
                self.assertGreater(result['jobs'][2]['dispatch_seconds'], result['jobs'][1]['dispatch_seconds'])
                self.assertEqual(len(result['selection']['calls']), 3)
                self.assertEqual(Path(result['answer_path']).read_text(), PROOF)
                for branch in result['branches']:
                    store = ProofStore.load(Path(branch['directory']) / 'workspace', branch['proof_id'])
                    self.assertEqual(store.state['settings']['max_seconds'], 4)
                # A failed first attempt cannot suppress the other logical
                # attempts or consume their individual wall-time allowances.
                result_dir = root / 'failed-first'
                fail_seed = branch_seed(7, 0)
                starts.clear()
                result = run_proof_portfolio(agent, 'Prove x=x for every real x.', output_dir=result_dir,
                    workers=3, branch_concurrency=1, max_tokens=60000, selection_tokens=6144,
                    max_seconds=15, selection_seconds=3, max_rounds=1, max_predict=512, seed=7)
                self.assertEqual([index for index, _, _ in starts], [0, 1, 2])
                self.assertTrue(all(seconds == 4 for _, _, seconds in starts))
                self.assertEqual(result['branches'][0]['proof_status'], 'paused')
                self.assertTrue(all(branch['candidate'] for branch in result['branches'][1:]))
                self.assertEqual(len(result['selection']['calls']), 2)
                self.assertEqual(maximum, 1)
                result_dir = root / 'over-cap-first'
                fail_seed, overspend_seed = None, branch_seed(7, 0)
                starts.clear()
                result = run_proof_portfolio(agent, 'Prove x=x for every real x.', output_dir=result_dir,
                    workers=3, branch_concurrency=1, max_tokens=60000, selection_tokens=6144,
                    max_seconds=15, selection_seconds=3, max_rounds=1, max_predict=512, seed=7)
                self.assertEqual([index for index, _, _ in starts], [0])
                self.assertEqual(result['status'], 'budget_violation')
                self.assertEqual(result['tokens_charged'], 257)
                self.assertTrue(result['branches'][0]['budget_violation'])
                self.assertTrue(all(branch['status'] == 'not_dispatched' for branch in result['branches'][1:]))
                self.assertNotIn('selection', result)
                self.assertIsNone(result['answer_path'])
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_wave_allocation_preserves_default_and_validates_concurrency(self):
        self.assertEqual(branch_schedule(3, None, 14400, 1800)['branch_seconds'], 12600)
        schedule = branch_schedule(3, 1, 43200, 3600)
        self.assertEqual(schedule['branch_seconds'], 13200)
        self.assertEqual(schedule['waves'], 3)
        self.assertEqual(branch_schedule(3, 2, 43200, 3600)['waves'], 2)
        self.assertEqual(branch_schedule(3, 8, 43200, 3600)['branch_concurrency'], 3)
        for width in (0, -1, 17, 1.5, True):
            with self.assertRaises(ValueError):
                branch_schedule(3, width, 43200, 3600)

    def test_two_processes_overlap_keep_sources_private_and_share_total_allocation(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        source = root / 'input'
        source.mkdir()
        (source / 'statement.txt').write_text('Prove x=x for every real x.')
        (source / 'reference.txt').write_text('PRIVATE_REFERENCE_SHOULD_NOT_BE_COPIED')
        lock = threading.Lock()
        requests, seeds = [], []
        overlap = threading.Barrier(2, timeout=10)
        overlapped = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with lock:
                    requests.append(payload)
                fields = payload.get('format', {}).get('properties', {})
                if not fields:
                    with lock:
                        seeds.append(payload['options']['seed'])
                    overlap.wait()
                    overlapped.set()
                    value = PROOF
                elif 'valid_steps' in fields:
                    value = {'valid_steps': ['Reflexivity holds.'], 'first_invalid_step': '',
                             'reason': '', 'missing_work': '', 'complete_candidate': True}
                elif 'claims' in fields:
                    claim = {'critical_claim': 'For every real x, x=x.', 'assumptions': ['x is real'],
                             'dependencies': [], 'argument': PROOF, 'disposition': 'supported',
                             'objection': '', 'evidence': '', 'next_task': 'Audit the complete proof.',
                             'new_progress': True, 'complete_candidate': False, 'resolves': [], 'resolution': ''}
                    value = {'claims': [claim], 'complete_candidate': True,
                             'next_task': 'Audit the complete proof.', 'strategy_summary': 'Apply reflexivity.'}
                elif 'first_invalid_step' in fields:
                    value = review()
                else:
                    value = {'verdict': 'complete', 'explanation': 'Reflexivity covers every stated x.',
                             'objection': '', 'next_task': ''}
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson')
                self.end_headers()
                self.wfile.write((json.dumps(event(value)) + '\n').encode())

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        client = Ollama(f'http://127.0.0.1:{server.server_port}')
        agent = Agent(client, Workspace(source), model='test', ctx=16384, predict=256)
        selector_goal = 'Prove x=x for every real x. This is the same complete benchmark goal in every arm.'
        result = run_proof_portfolio(agent, 'Prove the statement in statement.txt.',
            output_dir=root / 'portfolio', workers=2, max_tokens=10000, selection_tokens=1024,
            max_seconds=30, max_rounds=1, max_predict=512, source_files=['statement.txt'], seed=7,
            selector_goal=selector_goal, selection_seconds=4)
        self.assertEqual(result['selection_seconds'], 4)
        self.assertTrue(all(job['max_seconds'] == 26 for job in result['jobs']))
        self.assertTrue(all(job['request_timeout'] == client.timeout for job in result['jobs']))
        self.assertTrue(overlapped.is_set())
        self.assertEqual(sorted(seeds), [branch_seed(7, i) for i in range(2)])
        self.assertEqual(sum(j['token_budget'] for j in result['jobs']) + result['selection_tokens'], 10000)
        self.assertEqual(result['status'], 'selected_model_approved')
        self.assertEqual(result['tokens_charged'], 100)
        self.assertEqual(result['measured_completion_tokens'], 100)
        self.assertEqual(result['prompt_tokens'], 50)
        self.assertEqual(Path(result['answer_path']).read_text(), PROOF)
        self.assertNotIn('PRIVATE_REFERENCE_SHOULD_NOT_BE_COPIED', json.dumps(requests))
        raw_client = Client([[event(review())], [event(review())]])
        select_candidates(raw_client, model='test', goal=selector_goal,
            candidates=[{'id': str(i), 'text': PROOF, 'complete': True} for i in range(2)],
            ctx=16384, token_budget=1024, output_dir=root / 'raw-selection', seed=7)
        portfolio_reviews = [p for p in requests
            if 'first_invalid_step' in p.get('format', {}).get('properties', {})
            and 'valid_steps' not in p.get('format', {}).get('properties', {})]
        self.assertEqual(portfolio_reviews, raw_client.requests)
        for branch in result['branches']:
            workspace = Path(branch['directory']) / 'workspace'
            self.assertEqual((workspace / 'statement.txt').read_text(), (source / 'statement.txt').read_text())
            self.assertFalse((workspace / 'reference.txt').exists())
            with self.assertRaises(ValueError):
                Workspace(workspace).path('../../branch-001/workspace/statement.txt')
            store = ProofStore.load(workspace, branch['proof_id'])
            self.assertFalse(store.state['settings']['allow_literature'])
        prior_calls = len(requests)
        with self.assertRaises(FileExistsError):
            run_proof_portfolio(agent, 'Again', output_dir=root / 'portfolio')
        self.assertEqual(len(requests), prior_calls)

    def test_invalid_budget_rejected_before_creating_parent(self):
        with tempfile.TemporaryDirectory() as root:
            agent = Agent(Ollama(), Workspace(root))
            target = Path(root) / 'portfolio'
            with self.assertRaises(ValueError):
                run_proof_portfolio(agent, 'Prove it.', output_dir=target, workers=3,
                                    max_tokens=1000, selection_tokens=500)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
