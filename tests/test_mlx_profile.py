"""Transport checks for the explicit local MLX profile; no inference required."""
import copy
import io
import json
import contextlib
import os
import tempfile
import unittest
from unittest.mock import patch

from mathagent import cli
from mathagent.agent import AgentError
from mathagent.backends import OpenAICompatible
from mathagent.api_profiles import MLX, resolve_api_profile
from mathagent.gui.hub import Hub, Task
from test_backends import fixture, sse, delta, finish, usage


class MLXProfileTests(unittest.TestCase):
    def client(self, profile='mlx'):
        return (MLX if profile == 'mlx' else OpenAICompatible)('http://127.0.0.1:8766')

    def payload(self):
        return {'model': 'local-model', 'messages': [{'role': 'user', 'content': 'Test'}],
                'think': True, 'options': {'num_ctx': 32768, 'num_predict': 4096}}

    def test_context_and_thinking_match_the_mlx_api(self):
        wire = self.client()._payload(self.payload())
        self.assertEqual(wire['context_limit'], 32768)
        self.assertEqual(wire['chat_template_kwargs'], {'enable_thinking': True})
        self.assertEqual(wire['max_completion_tokens'], 4096)

    def test_default_profile_preserves_upstream_vllm_wire_contract(self):
        wire = self.client('default')._payload(self.payload())
        self.assertNotIn('context_limit', wire)
        self.assertTrue(wire['chat_template_kwargs']['preserve_thinking'])

    def test_tokenization_uses_matching_template_and_exact_server_counts(self):
        client = self.client()
        with patch.object(client, '_request_url', return_value=io.BytesIO(json.dumps(
            {'prompt_tokens': 31, 'max_context_tokens': 262144}).encode())) as send:
            self.assertEqual(client.count_input_tokens(self.payload()), 31)
        self.assertEqual(send.call_args.args[0], 'http://127.0.0.1:8766/api/tokenize')
        self.assertTrue(send.call_args.args[1]['thinking'])
        self.assertEqual(client.observed_context_tokens, 262144)

    def test_json_is_prompted_without_claiming_constrained_decoding_or_mutating_history(self):
        payload = self.payload()
        payload['format'] = {'type': 'object', 'required': ['status']}
        original = copy.deepcopy(payload)
        wire = self.client()._payload(payload)
        self.assertNotIn('structured_outputs', wire)
        self.assertNotIn('response_format', wire)
        self.assertIn('JSON schema', wire['messages'][-1]['content'])
        self.assertEqual(payload, original)

    def test_unsupported_seed_and_effort_fail_explicitly(self):
        payload = self.payload()
        payload['options']['seed'] = 1
        with self.assertRaisesRegex(AgentError, 'explicit seeds'):
            self.client()._payload(payload)
        payload['options'].pop('seed')
        payload['reasoning_effort'] = 'xhigh'
        with self.assertRaisesRegex(AgentError, 'reasoning_effort'):
            self.client()._payload(payload)

    def test_invalid_tokenizer_response_is_not_an_estimated_success(self):
        client = self.client()
        with patch.object(client, '_request_url', return_value=io.BytesIO(b'{"prompt_tokens":false,"max_context_tokens":32768}')):
            with self.assertRaisesRegex(AgentError, 'Malformed tokenizer'):
                client.count_input_tokens(self.payload())

    def test_optional_tokenizer_is_probed_once_and_preserves_proxy_prefix(self):
        client = MLX('http://localhost:8000/proxy/v1/')
        with patch.object(client, '_request_url', return_value=None) as send:
            self.assertIsNone(client.count_input_tokens(self.payload()))
            self.assertIsNone(client.count_input_tokens(self.payload()))
        send.assert_called_once()
        self.assertEqual(send.call_args.args[0], 'http://localhost:8000/proxy/api/tokenize')

    def test_requested_context_above_native_capacity_is_rejected(self):
        client = self.client()
        with patch.object(client, '_request_url', return_value=io.BytesIO(b'{"prompt_tokens":31,"max_context_tokens":8192}')):
            with self.assertRaisesRegex(AgentError, 'exceeds server context'):
                client.count_input_tokens(self.payload())

    def test_cli_profile_overrides_environment_and_invalid_profiles_fail(self):
        with patch.dict(os.environ, {'SQUARE_OPENAI_PROFILE': 'mlx'}):
            self.assertEqual(resolve_api_profile(), 'mlx')
            self.assertEqual(resolve_api_profile('default'), 'default')
        with self.assertRaisesRegex(AgentError, 'API profile'):
            resolve_api_profile('unknown')
        with patch.dict(os.environ, {'SQUARE_OPENAI_THINKING_FRACTION': '0.5'}):
            with self.assertRaisesRegex(AgentError, 'thinking-budget fractions'):
                self.client()

    def test_gui_profile_builds_a_pausable_mlx_client(self):
        with tempfile.TemporaryDirectory() as root:
            args = cli.parser().parse_args(['--backend', 'openai', '--api-profile', 'mlx',
                '--workspace', root, '--host', 'http://127.0.0.1:9', '--model', 'local-model'])
            hub, task = Hub(args), Task('proof', '', 'Proof')
            agent, _ = hub._engine(task)
            self.assertIsInstance(agent.client, MLX)
            self.assertEqual(agent.client.profile, 'mlx')
            task.cancel.set()
            with self.assertRaises(KeyboardInterrupt):
                next(agent.client.stream(self.payload()))

    def test_cli_chat_uses_mlx_over_real_http(self):
        tokenizer = json.dumps({'prompt_tokens': 100, 'max_context_tokens': 40960}).encode()
        with fixture(tokenizer, sse(delta(content='42'), finish(), usage())) as (base, requests, paths), \
                tempfile.TemporaryDirectory() as root, \
                patch('sys.argv', ['square-harness', '--backend', 'openai', '--api-profile', 'mlx',
                    '--host', base.host, '--workspace', root, '--model', 'square-qwen',
                    '--mode', 'critic', '--no-think', '--prompt', 'What is 19 plus 23?']), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(), 0)
            self.assertIn('42', out.getvalue())
            self.assertEqual(paths, ['/v1/models', '/api/tokenize', '/v1/chat/completions'])
            self.assertEqual(requests[-1]['chat_template_kwargs'], {'enable_thinking': False})
            self.assertEqual(requests[-1]['context_limit'], 40960)

    def test_mlx_profile_requires_openai_backend_before_contacting_server(self):
        with patch('sys.argv', ['square-harness', '--api-profile', 'mlx']), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            cli.main()
        self.assertEqual(caught.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
