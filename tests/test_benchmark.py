"""Private v0.5 trials preserve inputs, exact candidates and conservative costs."""
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mathagent import benchmark
from mathagent.agent import AgentError


class FakeClient:
    host, backend, timeout = 'http://127.0.0.1:8000', 'openai', 600

    def __init__(self, fail=False, usage=True):
        self.fail, self.usage, self.requests = fail, usage, []

    def models(self):
        return ['square-qwen']

    def stream(self, payload):
        self.requests.append(payload)
        if self.fail:
            raise AgentError('Simulated server failure')
        text = (json.dumps({'verdict': 'no_issue_found', 'explanation': 'Every step checks out.', 'issues': []})
                if payload.get('format') else 'For every real x, reflexivity gives x = x.')
        event = {'message': {'content': text}, 'done': True, 'done_reason': 'stop'}
        if self.usage:
            event.update(eval_count=11, prompt_eval_count=7)
        yield event


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = benchmark.init_smoke(self.root / 'dataset')
        self.args = benchmark.parser().parse_args(['--manifest', str(self.manifest), '--output', str(self.root / 'runs')])

    def test_manifest_is_allowlisted_and_rejects_answer_fields(self):
        self.assertEqual(len(benchmark.load_manifest(self.manifest)['problems']), 2)
        value = json.loads(self.manifest.read_text())
        value['problems'][0]['answer'] = 'Private reference answer.'
        self.manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'fields'):
            benchmark.load_manifest(self.manifest)

    def test_mutated_statement_fails_checksum(self):
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

    def test_dry_run_no_network_or_writes_and_only_two_arms(self):
        with patch.object(benchmark, 'create_client', side_effect=AssertionError('Network forbidden')):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = benchmark.main(['--manifest', str(self.manifest), '--output', str(self.root / 'runs'), '--dry-run'])
        self.assertEqual(code, 0)
        plan = json.loads(out.getvalue())
        self.assertEqual(plan['jobs'], 4)
        self.assertEqual(plan['settings']['arms'], ['raw-single', 'proof'])
        self.assertEqual(plan['generated_token_ceiling'], 2 * (32768 + 120000))
        self.assertFalse((self.root / 'runs').exists())

    def test_existing_output_and_output_in_dataset_rejected(self):
        self.args.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            benchmark.preflight(self.args)
        self.args.output = self.manifest.parent / 'results'
        with self.assertRaisesRegex(ValueError, 'outside'):
            benchmark.preflight(self.args)

    def test_resource_preflight_rejects_underfunded_cycle_and_concurrency(self):
        self.args.workers = 2
        with self.assertRaisesRegex(ValueError, 'workers'):
            benchmark.preflight(self.args)
        self.args.max_in_flight = 2
        benchmark.preflight(self.args)
        self.args.tokens = self.args.predict + self.args.verify_tokens - 1
        with self.assertRaisesRegex(ValueError, 'full solver and verifier'):
            benchmark.preflight(self.args)

    def test_context_and_invalid_time_guards_fail_before_dispatch(self):
        self.args.ctx = 16384
        with self.assertRaisesRegex(ValueError, 'Context'):
            benchmark.preflight(self.args)
        self.args.ctx = 40960
        for name in ('seconds', 'raw_seconds', 'request_timeout'):
            original = getattr(self.args, name)
            for value in (0, -1, float('nan'), float('inf')):
                setattr(self.args, name, value)
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, 'finite and positive'):
                    benchmark.preflight(self.args)
            setattr(self.args, name, original)

    def test_missing_usage_keeps_full_reservation_and_partial_text(self):
        result = benchmark._raw(FakeClient(usage=False), 'Prove x=x.', self.root / 'raw', self.args, 1000, 3)
        usage = benchmark._usage([result['call']])
        self.assertEqual(usage['tokens_charged'], 1000)
        self.assertEqual(usage['reserved_unmeasured_tokens'], 1000)
        self.assertTrue(Path(result['answer_path']).read_text())

    def test_runtime_context_rejection_never_dispatches_or_charges(self):
        class TooLarge(FakeClient):
            def count_input_tokens(self, payload):
                return self_ctx
        self_ctx = self.args.ctx
        client = TooLarge()
        result = benchmark._raw(client, 'Goal', self.root / 'raw', self.args, 32768, 0)
        self.assertEqual(client.requests, [])
        self.assertEqual(benchmark._usage([result['call']])['tokens_charged'], 0)
        self.assertEqual(result['call']['status'], 'error')

    def test_missing_completion_reason_is_not_success(self):
        class NoReason(FakeClient):
            def stream(self, payload):
                yield {'message': {'content': 'A candidate'}, 'done': True, 'eval_count': 3}
        result = benchmark._raw(NoReason(), 'Goal', self.root / 'raw', self.args, 1000, 0)
        self.assertFalse(result['complete'])
        self.assertEqual(result['call']['status'], 'error')

    def test_raw_budget_violation_retains_observed_usage(self):
        class OverspendingClient(FakeClient):
            def stream(self, payload):
                self.requests.append(payload)
                cap = payload['options']['num_predict']
                raise benchmark.TokenBudgetError({'eval_count': cap + 7, 'prompt_eval_count': 4}, cap)
        data, _ = benchmark.preflight(self.args)
        client = OverspendingClient()
        with patch.object(benchmark, 'create_client', return_value=client):
            result = benchmark.run_job((data['problems'][0], 'raw-single', 0, 4), self.root / 'runs', self.args)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['measured_completion_tokens'], self.args.predict + 7)
        self.assertEqual(result['reserved_unmeasured_tokens'], 0)
        self.assertEqual(len(client.requests), 1)

    def test_time_guard_closes_stream_and_preserves_answer(self):
        class SlowClient(FakeClient):
            closed = False
            def stream(self, payload):
                try:
                    yield {'message': {'content': 'Partial written argument.'}, 'done': False}
                    yield {'message': {}, 'done': True, 'eval_count': 5}
                finally:
                    self.closed = True
        client = SlowClient()
        with patch.object(benchmark.time, 'monotonic', side_effect=[0, 0, 2, 3]):
            result = benchmark._raw(client, 'Goal', self.root / 'raw', self.args, 1000, 0, max_seconds=1)
        self.assertTrue(client.closed)
        self.assertEqual(result['call']['status'], 'error')
        self.assertEqual(Path(result['answer_path']).read_text(), 'Partial written argument.')
        self.assertEqual(benchmark._usage([result['call']])['reserved_unmeasured_tokens'], 1000)

    def test_identical_initial_requests_and_all_proof_roles_charged(self):
        data, _ = benchmark.preflight(self.args)
        clients = [FakeClient(), FakeClient()]
        with patch.object(benchmark, 'create_client', side_effect=clients):
            raw = benchmark.run_job((data['problems'][0], 'raw-single', 0, 42), self.root / 'runs', self.args)
            proof = benchmark.run_job((data['problems'][0], 'proof', 0, 42), self.root / 'runs', self.args)
        self.assertEqual(clients[0].requests[0], clients[1].requests[0])
        self.assertTrue(clients[0].requests[0]['think'])
        self.assertNotIn('tools', clients[0].requests[0])
        self.assertEqual(proof['status'], 'candidate_complete', proof)
        self.assertEqual(proof['tokens_charged'], 22)
        self.assertEqual(proof['request_count'], 2)
        self.assertEqual(Path(raw['answer_path']).read_bytes(), Path(proof['answer_path']).read_bytes())

    def test_engine_selected_answer_is_exported_without_newest_draft_lookup(self):
        workspace = self.root / 'workspace'
        workspace.mkdir()
        agent = SimpleNamespace(workspace=SimpleNamespace(root=workspace))
        selected = {'id': 'candidate-1', 'artifact': 'original.md', 'transport_complete': True}
        state = {'status': 'budget_exhausted', 'pending': {'text': 'PARTIAL latest'}, 'calls': []}
        with patch.object(benchmark, 'ProofRunner') as runner, patch.object(benchmark, '_proof_usage', return_value=(benchmark._usage([]), [(workspace, state)])):
            runner.return_value.start.return_value = {'answer': 'Exact original proof.', 'status': 'budget_exhausted', 'selected_candidate': selected}
            outcome = benchmark._proof(agent, 'Original goal', self.root, self.args)
        self.assertEqual(Path(outcome['answer_path']).read_text(), 'Exact original proof.')
        self.assertEqual(outcome['selected_artifact'], 'original.md')
        self.assertEqual(runner.return_value.start.call_args.kwargs['source_files'], ())

    def test_failed_job_keeps_cost_and_does_not_destroy_other_results(self):
        data, _ = benchmark.preflight(self.args)
        with patch.object(benchmark, 'create_client', return_value=FakeClient(fail=True)):
            failed = benchmark.run_job((data['problems'][0], 'raw-single', 0, 11), self.root / 'runs', self.args)
        with patch.object(benchmark, 'create_client', return_value=FakeClient()):
            passed = benchmark.run_job((data['problems'][1], 'raw-single', 0, 12), self.root / 'runs', self.args)
        self.assertEqual(failed['status'], 'error')
        self.assertEqual(failed['reserved_unmeasured_tokens'], self.args.predict)
        self.assertEqual(passed['status'], 'complete')
        benchmark.export_grading(self.root / 'runs', [failed, passed], data['problems'], 7)
        sheet = (self.root / 'runs/grading/scores.csv').read_text()
        self.assertNotIn('raw-single', sheet)
        self.assertNotIn('model_approved', sheet)
        self.assertTrue((self.root / 'runs/grading-key.private.json').is_file())

    def test_execute_freezes_source_and_shared_seeds_without_grading(self):
        data, plan = benchmark.preflight(self.args)
        with patch.object(benchmark, 'create_client', side_effect=lambda *a, **k: FakeClient()), contextlib.redirect_stdout(io.StringIO()):
            records = benchmark.execute(self.args, data, plan)
        self.assertEqual(len(records), 4)
        for problem in data['problems']:
            self.assertEqual(len({r['seed'] for r in records if r['problem_id'] == problem['id']}), 1)
        self.assertEqual((self.args.output / 'source/mathagent/benchmark.py').read_bytes(), Path(benchmark.__file__).read_bytes())
        self.assertIn('No mathematical scores inferred', (self.args.output / 'summary.json').read_text())
        self.assertEqual(len((self.args.output / 'grading/scores.csv').read_text().splitlines()), 5)

    def test_source_drift_stops_before_network_or_output(self):
        data, plan = benchmark.preflight(self.args)
        plan['code_sha256'] = {}
        with patch.object(benchmark, 'create_client', side_effect=AssertionError('No network')):
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                benchmark.execute(self.args, data, plan)
        self.assertFalse(self.args.output.exists())

    def test_real_openai_http_cli_and_rerun_guard(self):
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
                if self.path.endswith('/tokenize'):
                    self.send_response(404)
                    self.end_headers()
                    return
                requests.append(payload)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                events = [
                    {'choices': [{'index': 0, 'delta': {'content': 'Synthetic response.'}, 'finish_reason': None}]},
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
                   '--host', f'http://127.0.0.1:{server.server_port}', '--workers', '2', '--max-in-flight', '2']
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=40)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            rows = [json.loads(line) for line in (self.root / 'http-run/outputs.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r['tokens_charged'] == 5 and r['prompt_tokens'] == 13 for r in rows))
            again = subprocess.run(command, text=True, capture_output=True, timeout=10)
            self.assertEqual(again.returncode, 2)
            self.assertIn('already exists', again.stderr)
            self.assertEqual(len(requests), 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
