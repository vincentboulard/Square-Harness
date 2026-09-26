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
from mathagent.backends import OpenAICompatible, TokenBudgetError, create_client
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


if __name__ == '__main__':
    unittest.main()
