import copy
import io
import json
import os
import runpy
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from mathagent.agent import Agent, AgentError, Ollama
from mathagent.backends import OpenAICompatible, LlamaCpp, TokenBudgetError, create_client
from mathagent.tools import Workspace


def delta(content=None, **fields):
    message = fields
    if content is not None:
        message['content'] = content
    return {'choices': [{'index': 0, 'delta': message, 'finish_reason': None}]}


def finish(reason='stop'):
    return {'choices': [{'index': 0, 'delta': {}, 'finish_reason': reason}]}


def usage(output=7, prompt=11):
    return {'choices': [], 'usage': {'prompt_tokens': prompt, 'completion_tokens': output,
                                   'completion_tokens_details': {'reasoning_tokens': 4}}}


def sse(*events, done=True):
    text = ': keepalive\r\n\r\n' + ''.join('data: ' + json.dumps(event, ensure_ascii=False) + '\r\n\r\n' for event in events)
    return (text + ('data: [DONE]\r\n\r\n' if done else '')).encode()


@contextmanager
def fixture(*responses, status=200, model_data=None):
    requests, paths, replies = [], [], list(responses)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            paths.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({'data': [{'id': 'square-qwen'}]} if model_data is None else model_data).encode())

        def do_POST(self):
            paths.append(self.path)
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(status)
            if status == 302:
                self.send_header('Location', '/unexpected-redirect')
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            body = replies.pop(0) if replies else b''
            if callable(body):
                body = body(requests[-1])
            try:
                # Deliberately split UTF-8 codepoints and SSE lines across writes.
                for index in range(0, len(body), 7):
                    self.wfile.write(body[index:index + 7])
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield OpenAICompatible(f'http://127.0.0.1:{server.server_port}', timeout=2), requests, paths
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@contextmanager
def llama_fixture(*responses, slot_context=32768, input_tokens=11, props=None,
                  count_response=None, count_status=200):
    requests, paths, replies = [], [], list(responses)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            paths.append(self.path)
            self.send_response(200)
            self.end_headers()
            value = ({'default_generation_settings': {'n_ctx': slot_context},
                      'chat_template': '{{ messages }}', 'total_slots': 3}
                     if props is None else props)
            if self.path.endswith('/models'):
                value = {'data': [{'id': 'square-qwen'}]}
            self.wfile.write(json.dumps(value).encode())

        def do_POST(self):
            paths.append(self.path)
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append((self.path, body))
            counting = self.path.endswith('/input_tokens')
            self.send_response(count_status if counting else 200)
            self.send_header('Content-Type', 'application/json' if counting else 'text/event-stream')
            self.end_headers()
            if counting:
                value = {'input_tokens': input_tokens} if count_response is None else count_response
                response = json.dumps(value).encode()
            else:
                response = replies.pop(0) if replies else b''
            try:
                self.wfile.write(response)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield LlamaCpp(f'http://127.0.0.1:{server.server_port}', timeout=2), requests, paths
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class BackendTests(unittest.TestCase):
    def payload(self, **kwargs):
        payload = {'model': 'square-qwen', 'messages': [{'role': 'user', 'content': 'Prove it.'}],
                   'think': True, 'options': {'num_ctx': 32768, 'num_predict': 128}}
        payload.update(kwargs)
        return payload

    def test_models_stream_thinking_and_usage_after_finish(self):
        body = sse(delta(reasoning='First '), delta(reasoning_content='verify. '), delta('Voilà.'), finish(), usage())
        with fixture(body) as (client, requests, paths):
            self.assertEqual(client.models(), ['square-qwen'])
            events = list(client.stream(self.payload()))
            self.assertEqual([event['message'].get('thinking') for event in events[:2]], ['First ', 'verify. '])
            self.assertEqual(events[-2]['message']['content'], 'Voilà.')
            # completion_tokens already includes reasoning; do not add it again.
            self.assertEqual(events[-1]['eval_count'], 7)
            self.assertEqual(events[-1]['prompt_eval_count'], 11)
            self.assertEqual(events[-1]['done_reason'], 'stop')
            self.assertEqual(paths, ['/v1/models', '/v1/chat/completions'])
            self.assertEqual(requests[0]['stream_options'], {'include_usage': True})

    def test_fragmented_tool_calls_and_real_agent_tool_round_trip(self):
        tool = sse(delta(reasoning='Inspect.'), delta(tool_calls=[{
            'index': 0, 'id': 'call_', 'type': 'function', 'function': {'name': 'read_', 'arguments': '{"pa'}}]),
            delta(tool_calls=[{'index': 0, 'id': 'abc', 'function': {'name': 'file', 'arguments': 'th":"lemma.txt"}'}}]),
            finish('tool_calls'), usage())
        with fixture(tool, sse(delta('A counterexample exists.'), finish(), usage())) as (client, requests, _):
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / 'lemma.txt').write_text('Every integer is even.')
                agent = Agent(client, Workspace(tmp), model='square-qwen', ctx=16384)
                self.assertEqual(agent.run('Read lemma.txt'), 'A counterexample exists.')
            history = requests[1]['messages']
            self.assertEqual(history[-2]['reasoning'], 'Inspect.')
            self.assertEqual(history[-2]['tool_calls'][0]['id'], 'call_abc')
            self.assertEqual(json.loads(history[-2]['tool_calls'][0]['function']['arguments']), {'path': 'lemma.txt'})
            self.assertEqual(history[-1]['tool_call_id'], 'call_abc')
            self.assertIn('Every integer is even.', history[-1]['content'])

    def test_interleaved_tool_calls_are_emitted_once_after_done(self):
        body = sse(delta(tool_calls=[{'index': 1, 'id': 'b', 'function': {'name': 'beta', 'arguments': '{}'}},
                                     {'index': 0, 'id': 'a', 'function': {'name': 'alpha', 'arguments': '{'}}]),
                   delta(tool_calls=[{'index': 0, 'function': {'arguments': '}'}}]), finish('tool_calls'), usage())
        with fixture(body) as (client, _, __):
            events = list(client.stream(self.payload()))
            self.assertEqual(len(events), 1)
            self.assertEqual([call['id'] for call in events[0]['message']['tool_calls']], ['a', 'b'])
            self.assertTrue(events[0]['done'])

    def test_options_schema_and_history_conversion_leave_payload_unchanged(self):
        payload = self.payload(think=False, reasoning_effort='low', format={'type': 'object', 'properties': {}})
        payload['options'].update(temperature=0.6, seed=42, top_p=.95, top_k=20, min_p=0, repeat_penalty=1)
        payload['messages'] += [{'role': 'assistant', 'content': '', 'thinking': 'Read.',
                                'tool_calls': [{'function': {'name': 'read_file', 'arguments': {'path': 'a'}}}]},
                                {'role': 'tool', 'tool_name': 'read_file', 'content': 'data'}]
        before = copy.deepcopy(payload)
        with fixture(sse(delta('{}'), finish(), usage())) as (client, requests, _):
            list(client.stream(payload))
            request = requests[0]
            self.assertEqual(request['max_completion_tokens'], 128)
            self.assertNotIn('num_ctx', request)
            self.assertNotIn('options', request)
            self.assertEqual(request['structured_outputs'], {'json': payload['format']})
            self.assertEqual(request['chat_template_kwargs'], {'enable_thinking': False, 'preserve_thinking': True})
            self.assertEqual(request['repetition_penalty'], 1)
            self.assertEqual(request['top_k'], 20)
            self.assertEqual(request['seed'], 42)
            self.assertEqual(request['reasoning_effort'], 'low')
            self.assertEqual(request['messages'][-1]['tool_call_id'], 'call_1_0')
        self.assertEqual(payload, before)

    def test_json_mode_and_length_finish(self):
        with fixture(sse(delta('{'), finish('length'), usage())) as (client, requests, _):
            events = list(client.stream(self.payload(format='json')))
            self.assertEqual(events[-1]['done_reason'], 'length')
            self.assertEqual(requests[0]['response_format'], {'type': 'json_object'})

    def test_length_during_tool_arguments_discards_calls_and_preserves_usage(self):
        body = sse(delta(tool_calls=[{'index': 0, 'id': 'call_a',
                                     'function': {'name': 'write_file', 'arguments': '{"path":'}}]),
                   finish('length'), usage(output=128))
        with fixture(body) as (client, _, __):
            events = list(client.stream(self.payload()))
            self.assertEqual(events[-1]['done_reason'], 'length')
            self.assertEqual(events[-1]['eval_count'], 128)
            self.assertFalse(any(event['message'].get('tool_calls') for event in events))

    def test_usage_above_output_cap_is_rejected(self):
        with fixture(sse(delta('bad accounting'), finish(), usage(output=129))) as (client, _, __):
            with self.assertRaisesRegex(TokenBudgetError, 'above the requested output cap') as caught:
                list(client.stream(self.payload()))
            self.assertEqual(caught.exception.stats, {'eval_count': 129, 'prompt_eval_count': 11})
            self.assertEqual(caught.exception.requested_cap, 128)

    def test_keepalives_do_not_bypass_elapsed_guard(self):
        client = OpenAICompatible(timeout=1)
        with patch.object(client, 'request', return_value=io.BytesIO(b': keepalive\n\n: keepalive\n\n')):
            with patch('mathagent.backends.time.monotonic', side_effect=[10, 10.5, 10.8, 11.1]):
                with self.assertRaisesRegex(AgentError, 'elapsed-time guard'):
                    list(client.stream(self.payload()))

    def test_multiline_sse_data_and_v1_base(self):
        body = b'data: {"choices":\ndata: [{"index":0,"delta":{"content":"ok"}}]}\n\n' + sse(finish(), usage())
        with fixture(body) as (client, _, paths):
            client = OpenAICompatible(client.host + '/v1/')
            events = list(client.stream(self.payload()))
            self.assertEqual(events[0]['message']['content'], 'ok')
            self.assertEqual(paths, ['/v1/chat/completions'])

    def test_missing_done_finish_or_usage_never_emit_completion(self):
        bodies = [sse(delta('partial'), finish(), usage(), done=False), sse(delta('partial'), usage()),
                  sse(delta('partial'), finish()), sse(delta('partial'), finish(), {'usage': {'completion_tokens': 7}})]
        for body in bodies:
            with self.subTest(body=body), fixture(body) as (client, _, __):
                events = []
                with self.assertRaises(AgentError):
                    events.extend(client.stream(self.payload()))
                self.assertFalse(any(event.get('done') for event in events))

    def test_malformed_calls_never_reach_tool_executor(self):
        for arguments in ('{"path":', '[]', 'null'):
            body = sse(delta(tool_calls=[{'index': 0, 'id': 'call_a', 'function': {'name': 'write_file', 'arguments': arguments}}]),
                       finish('tool_calls'), usage())
            with self.subTest(arguments=arguments), fixture(body) as (client, _, __):
                with self.assertRaises(AgentError):
                    list(client.stream(self.payload()))

    def test_errors_redirects_and_malformed_events(self):
        for body in (b'data: not json\n\n', sse({'error': {'message': 'failure'}}),
                     sse(finish('content_filter'), usage()), sse(finish('tool_calls'), usage())):
            with self.subTest(body=body), fixture(body) as (client, _, __):
                with self.assertRaises(AgentError):
                    list(client.stream(self.payload()))
        for status in (302, 400, 503):
            with self.subTest(status=status), fixture(b'failure', status=status) as (client, _, paths):
                with self.assertRaisesRegex(AgentError, f'HTTP {status}'):
                    list(client.stream(self.payload()))
                self.assertEqual(paths, ['/v1/chat/completions'])

    def test_ambient_proxy_not_used(self):
        with patch.dict(os.environ, {'HTTP_PROXY': 'http://127.0.0.1:1', 'http_proxy': 'http://127.0.0.1:1',
                                     'NO_PROXY': '', 'no_proxy': ''}):
            with fixture(sse(delta('ok'), finish(), usage())) as (client, _, __):
                self.assertEqual(list(client.stream(self.payload()))[0]['message']['content'], 'ok')

    def test_invalid_history_and_model_list(self):
        client = OpenAICompatible()
        for messages in ([{'role': 'tool', 'tool_name': 'missing', 'content': 'x'}],
                         [{'role': 'assistant', 'tool_calls': [{'function': {'name': 'read', 'arguments': {}}}]}]):
            with self.subTest(messages=messages), self.assertRaises(AgentError):
                list(client.stream(self.payload(messages=messages)))
        with fixture(model_data={'models': []}) as (client, _, __):
            with self.assertRaises(AgentError):
                client.models()

    def test_factory_and_invalid_hosts(self):
        self.assertIsInstance(create_client('ollama', 'http://localhost:11434'), Ollama)
        self.assertEqual(create_client('ollama', 'http://localhost:11434').backend, 'ollama')
        self.assertEqual(create_client('openai', 'http://localhost:8000', timeout=2).timeout, 2)
        for host in ('file:///tmp/model', 'http://user:secret@example.com', 'http://localhost:8000?key=secret'):
            with self.subTest(host=host), self.assertRaises(AgentError):
                OpenAICompatible(host)
        with self.assertRaises(AgentError):
            create_client('unknown', 'http://localhost')

    def test_live_smoke_script_exercises_local_server_and_concurrency(self):
        run_smoke = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts' / 'smoke-vllm.py'))['run_smoke']
        tool = sse(delta(tool_calls=[{'index': 0, 'id': 'nonce_call', 'function': {
            'name': 'read_smoke_nonce', 'arguments': '{}'}}]), finish('tool_calls'), usage())
        replies = [sse(delta('42'), finish(), usage()),
                   sse(delta(reasoning='Compute the product.'), delta('1517'), finish(), usage()), tool,
                   lambda request: sse(delta(request['messages'][-1]['content']), finish(), usage()),
                   sse(delta('{"status":"ok","count":2}'), finish(), usage()),
                   sse(delta('1'), finish('length'), usage(output=1))]
        replies.extend([sse(delta('Reproducible settings matter.'), finish(), usage())] * 7)
        with fixture(*replies) as (client, requests, paths):
            report = run_smoke(lambda: OpenAICompatible(client.host, timeout=2), 'square-qwen')
            self.assertTrue(report['ok'], report.get('error'))
            self.assertEqual([item['requests'] for item in report['concurrency']], [1, 2, 4])
            self.assertEqual([item['generated_tokens'] for item in report['concurrency']], [7, 14, 28])
            self.assertEqual(len(requests), 13)
            self.assertEqual(paths[0], '/v1/models')

    def test_live_smoke_failure_report_is_not_success(self):
        run_smoke = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts' / 'smoke-vllm.py'))['run_smoke']
        with fixture(sse(delta('wrong'), finish(), usage())) as (client, _, __):
            report = run_smoke(lambda: client, 'square-qwen')
            self.assertFalse(report['ok'])
            self.assertIn('Plain response did not return 42', report['error'])


class LlamaCppBackendTests(unittest.TestCase):
    def payload(self, **kwargs):
        payload = {'model': 'square-qwen', 'messages': [{'role': 'user', 'content': 'Prove it.'}],
                   'think': True, 'options': {'num_ctx': 32768, 'num_predict': 128}}
        payload.update(kwargs)
        return payload

    def test_native_wire_schema_exact_count_and_reasoning_usage(self):
        body = sse(delta(reasoning_content='Check the hypotheses.'), delta('{"status":"ok"}'), finish(), usage())
        payload = self.payload(think=False, format={'type': 'object', 'properties': {'status': {'type': 'string'}}})
        payload['options'].update(seed=42, top_k=20, min_p=0, repeat_penalty=1.05)
        original = copy.deepcopy(payload)
        with llama_fixture(body) as (client, requests, paths):
            events = list(client.stream(payload))
            self.assertEqual(paths, ['/props', '/v1/chat/completions/input_tokens', '/v1/chat/completions'])
            self.assertEqual(requests[0][1], requests[1][1])
            wire = requests[1][1]
            self.assertEqual(wire['max_tokens'], 128)
            self.assertNotIn('max_completion_tokens', wire)
            self.assertNotIn('structured_outputs', wire)
            self.assertEqual(wire['response_format']['json_schema']['schema'], payload['format'])
            self.assertEqual(wire['response_format']['type'], 'json_schema')
            self.assertEqual(wire['reasoning_format'], 'deepseek')
            self.assertEqual(wire['chat_template_kwargs'], {'enable_thinking': False, 'preserve_reasoning': True})
            self.assertEqual(wire['repeat_penalty'], 1.05)
            self.assertNotIn('repetition_penalty', wire)
            self.assertEqual(wire['seed'], 42)
            self.assertEqual(wire['min_p'], 0)
            self.assertEqual(events[0]['message']['thinking'], 'Check the hypotheses.')
            self.assertEqual(events[-1]['eval_count'], 7)  # Includes thinking; no double charge.
            self.assertEqual(events[-1]['prompt_eval_count'], 11)
        self.assertEqual(payload, original)

    def test_tool_roundtrip_preserves_reasoning_content_ids_and_arguments(self):
        tool = sse(delta(reasoning_content='Inspect the statement.'), delta(tool_calls=[{
            'index': 0, 'id': 'call_a', 'type': 'function',
            'function': {'name': 'read_file', 'arguments': '{"path":'}}]),
            delta(tool_calls=[{'index': 0, 'function': {'arguments': '"lemma.txt"}'}}]),
            finish('tool_calls'), usage())
        with llama_fixture(tool, sse(delta('Not every integer is even.'), finish(), usage())) as (client, requests, paths):
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / 'lemma.txt').write_text('Every integer is even.')
                agent = Agent(client, Workspace(tmp), model='square-qwen', ctx=16384)
                self.assertEqual(agent.run('Read lemma.txt'), 'Not every integer is even.')
            count_requests = [body for path, body in requests if path.endswith('/input_tokens')]
            generation_requests = [body for path, body in requests if path.endswith('/completions')]
            self.assertEqual(count_requests, generation_requests)
            history = generation_requests[1]['messages']
            self.assertEqual(history[-2]['reasoning_content'], 'Inspect the statement.')
            self.assertNotIn('reasoning', history[-2])
            self.assertEqual(history[-2]['tool_calls'][0]['id'], 'call_a')
            self.assertEqual(history[-1]['tool_call_id'], 'call_a')
            self.assertEqual(paths.count('/props'), 1)

    def test_per_slot_context_not_total_pool_is_enforced_before_generation(self):
        with llama_fixture(slot_context=8192) as (client, requests, paths):
            with self.assertRaisesRegex(AgentError, 'per-slot context 8192'):
                list(client.stream(self.payload()))
            self.assertEqual(paths, ['/props'])
            self.assertEqual(requests, [])

    def test_exact_prompt_plus_full_output_and_boundary_token_must_fit(self):
        for count, fits in ((871, True), (872, False)):
            with self.subTest(count=count), llama_fixture(sse(delta('ok'), finish(), usage(prompt=count)),
                                                         slot_context=1000, input_tokens=count) as (client, requests, _):
                payload = self.payload(options={'num_ctx': 1000, 'num_predict': 128})
                if fits:
                    self.assertTrue(list(client.stream(payload))[-1]['done'])
                    self.assertEqual(requests[-1][1]['max_tokens'], 128)
                else:
                    with self.assertRaisesRegex(AgentError, 'were not truncated'):
                        list(client.stream(payload))
                    self.assertEqual(len(requests), 1)
                    self.assertTrue(requests[0][0].endswith('/input_tokens'))

    def test_requested_context_is_honored_even_when_server_slot_is_larger(self):
        with llama_fixture(slot_context=32768, input_tokens=900) as (client, requests, _):
            with self.assertRaisesRegex(AgentError, 'Context budget exceeded'):
                list(client.stream(self.payload(options={'num_ctx': 1000, 'num_predict': 128})))
            self.assertEqual(len(requests), 1)

    def test_changed_actual_prompt_count_never_emits_completion_or_tools(self):
        with llama_fixture(sse(delta('partial'), finish(), usage(prompt=12)), input_tokens=11) as (client, _, __):
            events = []
            with self.assertRaisesRegex(AgentError, 'prompt token count changed'):
                events.extend(client.stream(self.payload()))
            self.assertFalse(any(event.get('done') for event in events))

    def test_output_cap_violation_retains_actual_usage(self):
        with llama_fixture(sse(delta('partial'), finish(), usage(output=129))) as (client, _, __):
            with self.assertRaises(TokenBudgetError) as caught:
                list(client.stream(self.payload()))
            self.assertEqual(caught.exception.requested_cap, 128)
            self.assertEqual(caught.exception.stats['eval_count'], 129)

    def test_count_endpoint_is_mandatory_and_malformed_counts_fail_closed(self):
        for response in ({}, {'input_tokens': True}, {'input_tokens': -1}, {'input_tokens': '11'}):
            with self.subTest(response=response), llama_fixture(count_response=response) as (client, requests, _):
                with self.assertRaises(AgentError):
                    list(client.stream(self.payload()))
                self.assertEqual(len(requests), 1)
        with llama_fixture(count_status=404) as (client, requests, _):
            with self.assertRaisesRegex(AgentError, 'HTTP 404'):
                list(client.stream(self.payload()))
            self.assertEqual(len(requests), 1)

    def test_invalid_server_properties_do_not_dispatch_generation(self):
        for props in ({}, {'default_generation_settings': None},
                      {'default_generation_settings': {'n_ctx': True}, 'chat_template': 'x'},
                      {'default_generation_settings': {'n_ctx': 32768}, 'chat_template': ''}):
            with self.subTest(props=props), llama_fixture(props=props) as (client, requests, _):
                with self.assertRaises(AgentError):
                    list(client.stream(self.payload()))
                self.assertEqual(requests, [])

    def test_factory_v1_host_count_helper_and_properties_cache_are_isolated(self):
        with llama_fixture(input_tokens=11) as (client, requests, paths):
            client = create_client('llamacpp', client.host + '/v1/', timeout=2)
            self.assertEqual(client.backend, 'llamacpp')
            self.assertEqual(client.models(), ['square-qwen'])
            props = client.properties()
            props['default_generation_settings']['n_ctx'] = 1
            self.assertEqual(client.properties()['default_generation_settings']['n_ctx'], 32768)
            self.assertEqual(client.count_input_tokens(self.payload()), 11)
            client.properties(refresh=True)
            self.assertEqual(paths.count('/props'), 2)
            self.assertEqual(requests[0][0], '/v1/chat/completions/input_tokens')

    def test_unsupported_effort_or_missing_context_or_cap_is_not_silently_ignored(self):
        client = LlamaCpp()
        for payload in (self.payload(reasoning_effort='xhigh'), self.payload(think='yes'),
                        self.payload(options={'num_predict': 128}), self.payload(options={'num_ctx': 1000})):
            with self.subTest(payload=payload), patch.object(client, 'request') as request:
                with self.assertRaises(AgentError):
                    list(client.stream(payload))
                request.assert_not_called()


if __name__ == '__main__':
    unittest.main()


class VllmContextCountTests(unittest.TestCase):
    def payload(self):
        return {'model': 'square-qwen', 'messages': [{'role': 'user', 'content': 'Check α=α.'}],
                'think': True, 'options': {'num_ctx': 40960, 'num_predict': 32768}}

    def test_exact_chat_count_preserves_template_and_proxy_prefix(self):
        response = json.dumps({'count': 21, 'max_model_len': 40960, 'tokens': [1] * 21}).encode()
        with fixture(response) as (client, requests, paths):
            client = OpenAICompatible(client.host + '/proxy/v1')
            self.assertEqual(client.count_input_tokens(self.payload()), 21)
            self.assertEqual(paths, ['/proxy/tokenize'])
            self.assertEqual(requests[0]['messages'], self.payload()['messages'])
            self.assertTrue(requests[0]['chat_template_kwargs']['enable_thinking'])
            self.assertTrue(requests[0]['add_generation_prompt'])
            self.assertFalse(requests[0]['add_special_tokens'])
            self.assertNotIn('max_completion_tokens', requests[0])

    def test_missing_endpoint_is_cached_as_unavailable_not_zero_tokens(self):
        for status in (404, 405):
            with self.subTest(status=status), fixture(b'not supported', status=status) as (client, requests, paths):
                self.assertIsNone(client.count_input_tokens(self.payload()))
                self.assertIsNone(client.count_input_tokens(self.payload()))
                self.assertEqual(len(requests), 1)

    def test_malformed_count_and_server_failure_do_not_bypass_context_guard(self):
        for value in ({'count': True, 'max_model_len': 40960}, {'count': -1, 'max_model_len': 40960},
                      {'count': 10}, {'count': 10, 'max_model_len': 0}):
            with self.subTest(value=value), fixture(json.dumps(value).encode()) as (client, _, __):
                with self.assertRaisesRegex(AgentError, 'Malformed tokenizer'):
                    client.count_input_tokens(self.payload())
        with fixture(b'failure', status=500) as (client, _, __):
            with self.assertRaisesRegex(AgentError, 'HTTP 500'):
                client.count_input_tokens(self.payload())

    def test_server_context_must_cover_requested_context(self):
        with fixture(json.dumps({'count': 10, 'max_model_len': 8192}).encode()) as (client, _, __):
            with self.assertRaisesRegex(AgentError, 'exceeds server context'):
                client.count_input_tokens(self.payload())
