"""Protocol/capacity acceptance tests with simulated inference, never GPU claims."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import runpy
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from mathagent.backends import create_client

SMOKE = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts' / 'smoke-serving.py'))
COMMIT = '2145525a4081d66ff1a87cf43ef809f95a85ac0c'
TEMPLATE = '{{ messages }}'


@contextmanager
def serving_fixture(*, serial=False, zero_decoded=False, context=4096, parallel=3,
                    wrong_json=False, change_identity=False, ready=True, model='square-qwen',
                    next_token_list=False, next_token_factory=None):
    active, requests = {}, []
    lock = threading.Lock()
    slots_gate = threading.Semaphore(1 if serial else parallel)

    def count(data):
        return len(json.dumps(data['messages'], ensure_ascii=False).split()) + 20

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def json_response(self, value, status=200):
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def do_GET(self):
            if self.path == '/health':
                self.json_response({'status': 'ok' if ready else 'loading model'}, 200 if ready else 503)
            elif self.path == '/props':
                self.json_response({'total_slots': parallel, 'default_generation_settings': {'n_ctx': context},
                    'model_path': '/models/model.gguf', 'build_info': 'b10000-' + ('abcdef0' if change_identity and requests else COMMIT[:7]),
                    'chat_template': TEMPLATE})
            elif self.path == '/slots':
                with lock:
                    values = [dict(active.get(i, {})) for i in range(parallel)]
                def next_token(item):
                    decoded = 0 if zero_decoded else item.get('decoded', 0)
                    if next_token_factory is not None:
                        return next_token_factory(decoded)
                    value = {'has_next_token': bool(item), 'has_new_line': False,
                             'n_remain': -1, 'n_decoded': decoded}
                    return [value] if next_token_list else value
                self.json_response([{'id': i, 'n_ctx': context, 'is_processing': bool(item),
                    'params': {'seed': item.get('seed')}, 'next_token': next_token(item)}
                    for i, item in enumerate(values)])
            elif self.path == '/v1/models':
                self.json_response({'data': [{'id': model}]})
            else:
                self.json_response({'error': 'unexpected endpoint'}, 404)

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path.endswith('/input_tokens'):
                self.json_response({'input_tokens': count(data)})
                return
            if self.path != '/v1/chat/completions':
                self.json_response({'error': 'unexpected endpoint'}, 404)
                return
            with lock:
                requests.append(data)
            text = data['messages'][0]['content']
            capacity = 'inference-capacity test' in text
            if capacity:
                slots_gate.acquire()
                with lock:
                    slot = next(i for i in range(parallel) if i not in active)
                    active[slot] = {'seed': data['seed'], 'decoded': 0}
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                def event(delta=None, finish=None, usage=None):
                    value = {'choices': [] if usage else [{'index': 0, 'delta': delta or {}, 'finish_reason': finish}]}
                    if usage:
                        value['usage'] = usage
                    self.wfile.write(('data: ' + json.dumps(value) + '\n\n').encode())
                    self.wfile.flush()
                reason, generated = 'stop', 7
                if capacity:
                    for i in range(5):
                        with lock:
                            active[slot]['decoded'] = i + 1
                        event({'content': str(i + 1) + ' '})
                        time.sleep(.025)
                    reason, generated = 'length', data['max_tokens']
                elif data.get('tools'):
                    event({'tool_calls': [{'index': 0, 'id': 'call_nonce', 'type': 'function',
                        'function': {'name': 'read_smoke_nonce', 'arguments': '{}'}}]})
                    reason = 'tool_calls'
                elif data['messages'][-1]['role'] == 'tool':
                    event({'content': data['messages'][-1]['content']})
                elif data.get('response_format'):
                    event({'content': json.dumps({'status': 'wrong' if wrong_json else 'ok', 'count': 2})})
                elif data['max_tokens'] == 1:
                    event({'content': '1'})
                    reason, generated = 'length', 1
                elif data.get('chat_template_kwargs', {}).get('enable_thinking'):
                    event({'reasoning_content': 'Compute the elementary product.'})
                    event({'content': '1517'})
                else:
                    event({'content': '42'})
                event(finish=reason)
                event(usage={'prompt_tokens': count(data), 'completion_tokens': generated, 'total_tokens': count(data) + generated})
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                if capacity:
                    with lock:
                        active.pop(slot, None)
                    slots_gate.release()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class ServingSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def launch(self, host, **overrides):
        value = {'schema_version': 1, 'host': host, 'model_alias': 'square-qwen',
                 'model_path': '/models/model.gguf', 'context_per_slot': 4096, 'parallel': 3,
                 'template_sha256': hashlib.sha256(TEMPLATE.encode()).hexdigest(),
                 'llama_commit': COMMIT, 'weights_sha256': 'a' * 64, 'image_id': 'sha256:test-image',
                 'context_shift': False, 'quantization': 'Q8_0'}
        value.update(overrides)
        path = self.root / 'launch.json'
        path.write_text(json.dumps(value))
        return path

    def run_smoke(self, host, model='square-qwen', **kwargs):
        options = dict(context=4096, parallel=3, representative_input_tokens=1024,
                       plain_tokens=128, thinking_tokens=512, capacity_output_tokens=32,
                       ready_timeout=.02, poll_interval=.005)
        options.update(kwargs)
        return SMOKE['run_acceptance'](lambda: create_client('llamacpp', host, timeout=2),
            SMOKE['Probe'](host, timeout=2), model, host=host, **options)

    def test_real_sse_protocol_context_capacity_overlap_and_launch_binding(self):
        with serving_fixture() as (host, requests):
            launch = self.launch(host)
            report = self.run_smoke(host, launch_record=launch)
        self.assertTrue(report['ok'], report)
        self.assertTrue(report['accepted_for_benchmark'])
        self.assertEqual(report['capacity_mode'], 'parallel')
        self.assertEqual(report['launch_record']['sha256'], hashlib.sha256(launch.read_bytes()).hexdigest())
        self.assertEqual(report['server']['slot_contexts'], [4096] * 3)
        self.assertEqual(len(requests), 12)  # six protocol generations, two batches of three
        for name in ('short_capacity', 'representative_capacity'):
            result = report['checks'][name]
            self.assertEqual(result['max_observed_decoding_slots'], 3)
            self.assertTrue(result['parallel_overlap_observed'])
            self.assertEqual(result['capacity_mode'], 'parallel')
            self.assertGreater(result['token_span_overlap_seconds'], 0)
            self.assertTrue(all(x >= result['target_input_tokens'] for x in result['observed_input_tokens']))
            self.assertTrue(all(r['generated_tokens'] <= 32 for r in result['requests_stats']))
        self.assertTrue(all('reasoning_effort' not in request for request in requests))
        self.assertTrue(report['checks']['protocol']['checks']['tool_round_trip'])

    def test_one_slot_q4_a10_cli_accepts_sequential_capacity_without_parallel_claim(self):
        alias = 'square-qwen-a10'
        output = self.root / 'a10-acceptance.json'
        with serving_fixture(context=32768, parallel=1, model=alias) as (host, requests):
            launch = self.launch(host, context_per_slot=32768, parallel=1,
                                 model_alias=alias, quantization='Q4_K_M')
            with redirect_stdout(io.StringIO()) as stdout:
                code = SMOKE['main'](['--host', host, '--model', alias, '--parallel', '1',
                    '--launch-record', str(launch), '--output', str(output), '--timeout', '2',
                    '--ready-timeout', '.02', '--poll-interval', '.005', '--capacity-output-tokens', '32'])
        self.assertEqual(code, 0, stdout.getvalue())
        report = json.loads(output.read_text())
        self.assertTrue(report['accepted_for_benchmark'], report)
        self.assertEqual(report['capacity_mode'], 'sequential')
        self.assertEqual(report['acceptance_condition'], 'one_slot_streamed_decoding')
        self.assertEqual(report['settings']['parallel'], 1)
        self.assertEqual(report['settings']['representative_input_tokens'], 30000)
        self.assertEqual(report['launch_record']['quantization'], 'Q4_K_M')
        self.assertEqual(len(requests), 8)
        self.assertTrue(all(request['model'] == alias for request in requests))
        for name in ('short_capacity', 'representative_capacity'):
            result = report['checks'][name]
            self.assertTrue(result['ok'])
            self.assertEqual(result['capacity_mode'], 'sequential')
            self.assertEqual(result['max_observed_decoding_slots'], 1)
            self.assertFalse(result['parallel_overlap_observed'])
            self.assertIsNone(result['token_span_overlap_seconds'])
            self.assertGreater(result['token_span_seconds'], 0)
            self.assertIn('Parallel inference was not tested', result['scope'])
        self.assertGreaterEqual(report['checks']['representative_capacity']['observed_input_tokens'][0], 30000)

    def test_pinned_slot_array_metadata_accepts_one_and_three_streaming_slots(self):
        for parallel in (1, 3):
            with self.subTest(parallel=parallel):
                with serving_fixture(parallel=parallel, next_token_list=True) as (host, requests):
                    report = self.run_smoke(host, parallel=parallel,
                                            launch_record=self.launch(host, parallel=parallel))
                self.assertTrue(report['accepted_for_benchmark'], report)
                self.assertEqual(len(requests), 6 + 2 * parallel)
                for name in ('short_capacity', 'representative_capacity'):
                    capacity = report['checks'][name]
                    self.assertEqual(capacity['max_observed_decoding_slots'], parallel)
                    self.assertEqual(capacity['parallel_overlap_observed'], parallel > 1)
                    self.assertTrue(any(len(sample['processing_slots']) == parallel
                        and all(slot['n_decoded'] > 0 for slot in sample['processing_slots'])
                        for sample in capacity['slot_samples']))

    def test_empty_or_malformed_slot_array_is_rejected_over_http(self):
        for metadata in ([], [None], [{'n_decoded': True}], [{'n_decoded': 1}, {}]):
            with self.subTest(metadata=metadata):
                with serving_fixture(parallel=1, next_token_factory=lambda _, value=metadata: value) as (host, _):
                    report = self.run_smoke(host, parallel=1,
                                            launch_record=self.launch(host, parallel=1))
                self.assertFalse(report['ok'])
                self.assertFalse(report['accepted_for_benchmark'])
                self.assertIn('Active slot next_token', report['error'])

    def test_slot_decode_metadata_rejects_invalid_counts(self):
        for count in (None, True, -1, 1.5, '2'):
            for metadata in ({'n_decoded': count}, [{'n_decoded': count}]):
                with self.subTest(metadata=metadata), self.assertRaises(SMOKE['SmokeFailure']):
                    SMOKE['slot_decoded_tokens']({'next_token': metadata})

    def test_array_slot_metadata_with_no_decoding_cannot_prove_capacity(self):
        with serving_fixture(parallel=1, next_token_list=True, zero_decoded=True) as (host, _):
            report = self.run_smoke(host, parallel=1, launch_record=self.launch(host, parallel=1))
        self.assertFalse(report['accepted_for_benchmark'])
        self.assertIn('No actively decoding slot', report['error'])

    def test_one_slot_wrong_launch_parallel_is_rejected_before_inference(self):
        alias = 'square-qwen-a10'
        with serving_fixture(parallel=1, model=alias) as (host, requests):
            report = self.run_smoke(host, model=alias, parallel=1,
                                   launch_record=self.launch(host, parallel=3, model_alias=alias, quantization='Q4_K_M'))
        self.assertFalse(report['ok'])
        self.assertFalse(report['accepted_for_benchmark'])
        self.assertIn('Launch context/parallel settings differ', report['error'])
        self.assertEqual(requests, [])

    def test_one_slot_without_observed_decoding_is_not_accepted(self):
        with serving_fixture(parallel=1, zero_decoded=True) as (host, _):
            report = self.run_smoke(host, parallel=1, launch_record=self.launch(host, parallel=1))
        self.assertFalse(report['ok'])
        self.assertIn('No actively decoding slot', report['error'])
        self.assertFalse(report['checks']['short_capacity']['parallel_overlap_observed'])

    def test_concurrent_submissions_with_serial_processing_are_rejected(self):
        with serving_fixture(serial=True) as (host, requests):
            report = self.run_smoke(host, launch_record=self.launch(host))
        self.assertFalse(report['ok'])
        self.assertFalse(report['accepted_for_benchmark'])
        self.assertIn('simultaneously decoding', report['error'])
        self.assertEqual(len(requests), 9)

    def test_processing_slots_without_decoded_tokens_do_not_prove_overlap(self):
        with serving_fixture(zero_decoded=True) as (host, _):
            report = self.run_smoke(host, launch_record=self.launch(host))
        self.assertFalse(report['ok'])
        self.assertIn('0/3 simultaneously decoding', report['error'])

    def test_insufficient_per_slot_context_fails_before_inference(self):
        with serving_fixture(context=2048) as (host, requests):
            report = self.run_smoke(host)
        self.assertFalse(report['ok'])
        self.assertIn('per-slot context', report['error'])
        self.assertEqual(requests, [])

    def test_launch_identity_mismatch_fails_before_inference(self):
        with serving_fixture() as (host, requests):
            report = self.run_smoke(host, launch_record=self.launch(host, model_path='/models/other.gguf'))
        self.assertFalse(report['ok'])
        self.assertIn('model path differs', report['error'])
        self.assertEqual(requests, [])

    def test_protocol_success_without_launch_record_cannot_authorize_benchmark(self):
        with serving_fixture() as (host, _):
            report = self.run_smoke(host)
        self.assertTrue(report['ok'], report)
        self.assertFalse(report['accepted_for_benchmark'])
        self.assertFalse(report['launch_record']['verified'])

    def test_structured_output_failure_stops_before_capacity(self):
        with serving_fixture(wrong_json=True) as (host, requests):
            report = self.run_smoke(host, launch_record=self.launch(host))
        self.assertFalse(report['ok'])
        self.assertIn('Structured output violates', report['error'])
        self.assertNotIn('short_capacity', report['checks'])
        self.assertLess(len(requests), 7)

    def test_server_identity_change_during_smoke_is_rejected(self):
        with serving_fixture(change_identity=True) as (host, _):
            report = self.run_smoke(host, launch_record=self.launch(host))
        self.assertFalse(report['ok'])
        self.assertIn('configuration changed', report['error'])

    def test_readiness_failure_is_bounded_and_no_inference_occurs(self):
        with serving_fixture(ready=False) as (host, requests):
            report = self.run_smoke(host)
        self.assertFalse(report['ok'])
        self.assertIn('Readiness deadline', report['error'])
        self.assertEqual(requests, [])

    def test_existing_report_is_refused_before_any_client_creation(self):
        path = self.root / 'existing.json'
        path.write_text('{}')
        with patch.dict(SMOKE['main'].__globals__, create_client=lambda *args, **kwargs: self.fail('Must not create a client')):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                SMOKE['main'](['--output', str(path)])
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(path.read_text(), '{}')


if __name__ == '__main__':
    unittest.main()
