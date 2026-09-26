import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mathagent import benchmark
from mathagent.agent import AgentError
from mathagent.portfolio import branch_seed


class FakeClient:
    host, backend, timeout = 'http://127.0.0.1:8000', 'openai', 600

    def __init__(self, fail=False, usage=True):
        self.fail, self.usage, self.requests = fail, usage, []

    def stream(self, payload):
        self.requests.append(payload)
        if self.fail:
            raise AgentError('Simulated server failure')
        if payload.get('format'):
            text = json.dumps({'verdict': 'gap', 'explanation': 'The proposed argument has an unchecked implication.',
                               'first_invalid_step': 'The induction step is missing.'})
        else:
            text = 'A full candidate text, with a potentially incomplete mathematical argument.'
        event = {'message': {'content': text}, 'done': True, 'done_reason': 'stop'}
        if self.usage:
            event.update(eval_count=11, prompt_eval_count=7)
        yield event


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = benchmark.init_smoke(self.root / 'dataset')
        self.args = benchmark.parser().parse_args(['--manifest', str(self.manifest), '--output', str(self.root / 'runs')])

    def tearDown(self):
        self.temp.cleanup()

    def test_manifest_is_allowlisted_hashes_inputs_and_rejects_answer_fields(self):
        data = benchmark.load_manifest(self.manifest)
        self.assertEqual(len(data['problems']), 2)
        original = json.loads(self.manifest.read_text())
        original['problems'][0]['answer'] = 'Do not send this to a solver.'
        self.manifest.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, 'fields'):
            benchmark.load_manifest(self.manifest)

    def test_mutated_statement_fails_declared_checksum(self):
        (self.manifest.parent / 'toy_identity.txt').write_text('Different problem')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            benchmark.load_manifest(self.manifest)

    def test_traversal_and_symlink_escape_rejected(self):
        value = json.loads(self.manifest.read_text())
        value['problems'][0] = {'id': 'one', 'statement': '../outside.txt'}
        self.manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'traversal'):
            benchmark.load_manifest(self.manifest)
        (self.root / 'outside.txt').write_text('Hidden answer')
        (self.manifest.parent / 'link.txt').symlink_to(self.root / 'outside.txt')
        value['problems'][0]['statement'] = 'link.txt'
        self.manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'contained'):
            benchmark.load_manifest(self.manifest)

    def test_dry_run_has_no_network_and_no_writes(self):
        with patch.object(benchmark, 'create_client', side_effect=AssertionError('Network/client creation forbidden')):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = benchmark.main(['--manifest', str(self.manifest), '--output', str(self.root / 'runs'), '--dry-run'])
        self.assertEqual(code, 0)
        plan = json.loads(out.getvalue())
        self.assertEqual(plan['jobs'], 6)
        self.assertEqual(plan['generated_token_ceiling'], 360000)
        self.assertFalse((self.root / 'runs').exists())

    def test_existing_output_and_output_in_dataset_fail_closed(self):
        self.args.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            benchmark.preflight(self.args)
        self.args.output = self.manifest.parent / 'results'
        with self.assertRaisesRegex(ValueError, 'outside'):
            benchmark.preflight(self.args)

    def test_workers_times_branch_width_is_bounded(self):
        self.args.workers = 2
        with self.assertRaisesRegex(ValueError, 'branch width'):
            benchmark.preflight(self.args)
        self.args.max_in_flight = 6
        benchmark.preflight(self.args)

    def test_context_check_includes_raw_partition_not_just_predict(self):
        self.args.ctx = 16384
        with self.assertRaisesRegex(ValueError, 'Context'):
            benchmark.preflight(self.args)

    def test_v100_time_guards_are_recorded_and_forwarded_to_direct_requests(self):
        self.args.backend = 'llamacpp'
        self.args.seconds = 14400
        self.args.request_timeout = 7200
        self.args.selection_seconds = 1800
        data, plan = benchmark.preflight(self.args)
        self.assertEqual(plan['settings']['request_timeout'], 7200)
        self.assertEqual(plan['settings']['selection_seconds'], 1800)
        output = self.root / 'runs'
        output.mkdir()
        with patch.object(benchmark, 'create_client', return_value=FakeClient()) as factory:
            benchmark.run_job((data['problems'][0], 'raw-best', 0, 4), output, self.args, None)
        self.assertEqual(len(factory.call_args_list), 4)
        self.assertTrue(all(c.kwargs['timeout'] == 7200 for c in factory.call_args_list))

    def test_invalid_request_and_selection_time_guards_fail_before_dispatch(self):
        for field in ('request_timeout', 'selection_seconds'):
            for value in (0, -1, float('nan'), float('inf')):
                with self.subTest(field=field, value=value):
                    args = benchmark.parser().parse_args([
                        '--manifest', str(self.manifest), '--output', str(self.root / 'runs')])
                    setattr(args, field, value)
                    with self.assertRaisesRegex(ValueError, 'finite and positive'):
                        benchmark.preflight(args)

    def test_missing_usage_keeps_full_reservation_and_partial_text(self):
        client = FakeClient(usage=False)
        result = benchmark._raw(client, 'Prove the goal.', self.root / 'raw', self.args, 1000, 3)
        usage = benchmark._usage([result['call']])
        self.assertEqual(usage['tokens_charged'], 1000)
        self.assertEqual(usage['reserved_unmeasured_tokens'], 1000)
        self.assertEqual(usage['measured_completion_tokens'], 0)
        self.assertTrue(Path(result['answer_path']).read_text())

    def test_raw_budget_violation_retains_observed_usage_and_stops_selection(self):
        class OverspendingClient(FakeClient):
            def stream(self, payload):
                self.requests.append(payload)
                cap = payload['options']['num_predict']
                raise benchmark.TokenBudgetError({'eval_count': cap + 7, 'prompt_eval_count': 4}, cap)
        data, _ = benchmark.preflight(self.args)
        output = self.root / 'runs'
        output.mkdir()
        client = OverspendingClient()
        with patch.object(benchmark, 'create_client', return_value=client):
            result = benchmark.run_job((data['problems'][0], 'raw-best', 0, 4), output, self.args, None)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(len(client.requests), 3)
        self.assertTrue(all('format' not in request for request in client.requests))
        self.assertEqual(result['measured_completion_tokens'], 3 * (17952 + 7))
        self.assertEqual(result['reserved_unmeasured_tokens'], 0)
        self.assertEqual(result['prompt_tokens'], 12)

    def test_raw_time_guard_closes_stream_and_preserves_partial_answer(self):
        class SlowClient(FakeClient):
            closed = False
            def stream(self, payload):
                try:
                    yield {'message': {'content': 'Partial written argument.'}, 'done': False}
                    yield {'message': {}, 'done': True, 'eval_count': 5}
                finally:
                    self.closed = True
        client = SlowClient()
        with patch.object(benchmark.time, 'monotonic', side_effect=[0, 2, 3]):
            result = benchmark._raw(client, 'Goal', self.root / 'raw', self.args, 1000, 0, max_seconds=1)
        self.assertTrue(client.closed)
        self.assertEqual(result['call']['status'], 'error')
        self.assertEqual(Path(result['answer_path']).read_text(), 'Partial written argument.')
        self.assertEqual(benchmark._usage([result['call']])['reserved_unmeasured_tokens'], 1000)

    def test_raw_best_uses_same_branch_seeds_shared_selector_and_exact_answer(self):
        data, _ = benchmark.preflight(self.args)
        output = self.root / 'runs'
        output.mkdir()
        clients = []
        def factory(*args, **kwargs):
            client = FakeClient()
            clients.append(client)
            return client
        seed = 17
        with patch.object(benchmark, 'create_client', side_effect=factory):
            result = benchmark.run_job((data['problems'][0], 'raw-best', 0, seed), output, self.args, threading.BoundedSemaphore(3))
        self.assertEqual(result['status'], 'selected_unverified')
        self.assertEqual(result['tokens_charged'], 66)
        self.assertEqual(result['prompt_tokens'], 42)
        requests = [request for client in clients for request in client.requests]
        raw = [r for r in requests if not r.get('format')]
        review = [r for r in requests if r.get('format')]
        self.assertEqual(sorted(r['options']['seed'] for r in raw), sorted(branch_seed(seed, i) for i in range(3)))
        self.assertTrue(all(r['options']['temperature'] == 0 for r in review))
        self.assertTrue(all(r['options']['num_predict'] == 17952 for r in raw))
        self.assertTrue(all('tools' not in r for r in raw))
        self.assertEqual(Path(result['answer_path']).read_bytes(), Path(result['raw_at_1_path']).read_bytes())
        self.assertEqual(result['answer_sha256'], result['raw_at_1_sha256'])
        self.assertEqual(list((output / 'jobs' / result['id'] / 'workspace').iterdir())[0].name, 'statement.txt')

    def test_failed_job_does_not_destroy_other_results_or_hide_charge(self):
        data, _ = benchmark.preflight(self.args)
        output = self.root / 'runs'
        output.mkdir()
        with patch.object(benchmark, 'create_client', return_value=FakeClient(fail=True)):
            failed = benchmark.run_job((data['problems'][0], 'raw-single', 0, 11), output, self.args, threading.BoundedSemaphore(1))
        with patch.object(benchmark, 'create_client', return_value=FakeClient()):
            passed = benchmark.run_job((data['problems'][1], 'raw-single', 0, 12), output, self.args, threading.BoundedSemaphore(1))
        self.assertEqual(failed['status'], 'error')
        self.assertEqual(failed['reserved_unmeasured_tokens'], self.args.predict)
        self.assertEqual(passed['status'], 'complete')
        benchmark.export_grading(output, [failed, passed], data['problems'], 7)
        sheet = (output / 'grading' / 'scores.csv').read_text()
        self.assertNotIn('raw-single', sheet)
        self.assertNotIn('model_approved', sheet)
        self.assertEqual(sheet.count('submission-'), 6)  # three appearances per row
        self.assertTrue((output / 'grading-key.private.json').is_file())

    def test_rejected_old_audit_does_not_replace_newer_draft(self):
        directory = self.root / 'proof'
        (directory / 'artifacts').mkdir(parents=True)
        (directory / 'artifacts' / 'old.json').write_text(json.dumps({'text': 'Rejected old argument', 'complete': True}))
        (directory / 'artifacts' / 'new.json').write_text(json.dumps({'text': 'Newest written candidate', 'complete': True}))
        workspace = self.root / 'workspace'
        workspace.mkdir()
        agent = SimpleNamespace(workspace=SimpleNamespace(root=workspace))
        state = {'status': 'budget_exhausted', 'final_audit': {'candidate': 'old.json', 'verdict': 'gap'},
                 'pending': {'draft': 'new.json'}, 'rounds': [{'draft': 'old.json'}], 'calls': []}
        with patch.object(benchmark, 'ProofRunner') as runner, patch.object(benchmark, '_proof_usage', return_value=(benchmark._usage([]), [(directory, state)])):
            runner.return_value.start.return_value = {'status': 'budget_exhausted', 'directory': str(directory)}
            outcome = benchmark._sequential(agent, 'Original goal', self.root, self.args)
        self.assertEqual(outcome['selected_artifact'], 'new.json')
        self.assertEqual(Path(outcome['answer_path']).read_text(), 'Newest written candidate')
        state['pending'] = None
        state['rounds'].append({'draft': 'new.json'})
        with patch.object(benchmark, 'ProofRunner') as runner, patch.object(benchmark, '_proof_usage', return_value=(benchmark._usage([]), [(directory, state)])):
            runner.return_value.start.return_value = {'status': 'budget_exhausted', 'directory': str(directory)}
            outcome = benchmark._sequential(agent, 'Original goal', self.root, self.args)
        self.assertEqual(outcome['selected_artifact'], 'new.json')

    def test_paused_transport_failure_is_benchmark_error_not_success(self):
        data, _ = benchmark.preflight(self.args)
        output = self.root / 'runs'
        output.mkdir()
        with patch.object(benchmark, 'create_client', return_value=FakeClient(fail=True)):
            result = benchmark.run_job((data['problems'][0], 'sequential', 0, 8), output, self.args, None)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['proof_status'], 'paused')
        self.assertIn('Simulated server failure', result['error'])
        self.assertEqual(result['reserved_unmeasured_tokens'], self.args.predict)
        self.assertEqual(Path(result['answer_path']).read_text(), '')

    def test_failed_parallel_branches_preserve_error_denominator_and_export(self):
        data, _ = benchmark.preflight(self.args)
        output = self.root / 'runs'
        output.mkdir()
        failed = {'status': 'no_complete_candidate', 'answer_path': None, 'tokens_charged': 2000,
                  'measured_completion_tokens': 0, 'reserved_unmeasured_tokens': 2000, 'prompt_tokens': 0,
                  'branches': [{'status': 'finished', 'proof_status': 'paused', 'calls': [{'status': 'interrupted'}]}],
                  'selection': {'calls': []}}
        with patch.object(benchmark, 'create_client', return_value=FakeClient()), patch('mathagent.portfolio.run_proof_portfolio', return_value=failed):
            result = benchmark.run_job((data['problems'][0], 'parallel', 0, 8), output, self.args, None)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['workflow_status'], 'no_complete_candidate')
        self.assertEqual(result['request_count'], 1)
        self.assertEqual(result['reserved_unmeasured_tokens'], 2000)
        benchmark.export_grading(output, [result], data['problems'], 7)
        self.assertTrue((output / 'grading' / 'submission-0001' / 'answer.md').is_file())

    def test_real_openai_http_cli_records_two_jobs_and_rejects_rerun(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"data":[{"id":"test-model"}]}')
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(payload)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
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
                schema = payload.get('structured_outputs', {}).get('json')
                text = json.dumps(value(schema)) if schema else 'Synthetic test response.'
                events = [
                    {'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]},
                    {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]},
                    {'choices': [], 'usage': {'prompt_tokens': 13, 'completion_tokens': 5, 'total_tokens': 18}},
                ]
                for event in events:
                    self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
                self.wfile.write(b'data: [DONE]\n\n')
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        command = [sys.executable, '-m', 'mathagent.benchmark', '--manifest', str(self.manifest),
                   '--output', str(self.root / 'http-run'), '--arms', 'raw-single', '--model', 'test-model',
                   '--host', f'http://127.0.0.1:{server.server_port}', '--workers', '2']
        try:
            process = subprocess.run(command, text=True, capture_output=True, timeout=40)
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            rows = [json.loads(line) for line in (self.root / 'http-run' / 'outputs.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r['tokens_charged'] == 5 and r['prompt_tokens'] == 13 for r in rows))
            self.assertTrue(all(Path(r['answer_path']).read_text() == 'Synthetic test response.' for r in rows))
            self.assertEqual(len(requests), 2)
            again = subprocess.run(command, text=True, capture_output=True, timeout=10)
            self.assertEqual(again.returncode, 2)
            self.assertIn('already exists', again.stderr)
            self.assertEqual(len(requests), 2)
            all_arms = [sys.executable, '-m', 'mathagent.benchmark', '--manifest', str(self.manifest),
                        '--output', str(self.root / 'all-arms'), '--model', 'test-model',
                        '--host', f'http://127.0.0.1:{server.server_port}', '--rounds', '1',
                        '--ctx', '16384', '--predict', '1024', '--max-predict', '1024',
                        '--tokens', '12000', '--selection-tokens', '768', '--branches', '2']
            process = subprocess.run(all_arms, text=True, capture_output=True, timeout=40)
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
            rows = [json.loads(line) for line in (self.root / 'all-arms' / 'outputs.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual({r['arm'] for r in rows}, {'raw-best', 'sequential', 'parallel'})
            self.assertTrue(all(r['status'] != 'error' for r in rows), rows)
            self.assertTrue(all(r['measured_completion_tokens'] > 0 for r in rows))
            self.assertTrue(all(Path(r['answer_path']).read_text() == 'Synthetic test response.' for r in rows))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
