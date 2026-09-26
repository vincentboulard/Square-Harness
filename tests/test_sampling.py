"""Saved settings prevent backend switches and free sampling resets on resume."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mathagent.agent import Agent, AgentError
from mathagent.backends import TokenBudgetError
from mathagent import cli
from mathagent.proof import ProofRunner
from mathagent.tools import Workspace


class InterruptedClient:
    host = 'http://localhost:8000'
    backend = 'openai'
    timeout = 10

    def __init__(self):
        self.requests = []

    def stream(self, payload):
        self.requests.append(payload)
        raise AgentError('Simulated interrupted request')
        yield  # stream is a generator, like the real transports


class SamplingTests(unittest.TestCase):
    def test_server_budget_violation_records_usage_and_prevents_resume_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = InterruptedClient()
            stats = {'eval_count': 513, 'prompt_eval_count': 77}
            client.stream = Mock(side_effect=TokenBudgetError(stats, 512))
            agent = Agent(client, Workspace(temporary), ctx=16384, predict=512)
            runner = ProofRunner(agent)
            result = runner.start('Prove that x=x.', max_tokens=8000)
            self.assertEqual(result['status'], 'budget_violation')
            self.assertEqual(runner.state['tokens_charged'], 513)
            self.assertEqual(runner.state['calls'][0]['stats'], stats)
            self.assertEqual(ProofRunner(agent).resume(result['id'])['status'], 'budget_violation')
            client.stream.assert_called_once()

    def test_resume_restores_sampling_and_advances_seed_after_charged_interruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = InterruptedClient()
            agent = Agent(client, Workspace(temporary), ctx=16384, predict=512,
                          seed=42, temperature=0.4, top_p=0.9)
            runner = ProofRunner(agent)
            result = runner.start('Prove that x=x for every real x.', max_tokens=8000)
            self.assertEqual(result['status'], 'paused')
            self.assertEqual(client.requests[0]['options']['seed'], 42)
            self.assertEqual(client.requests[0]['options']['temperature'], 0.4)
            charged = runner.state['tokens_charged']
            agent.seed, agent.temperature, agent.top_p = 100, 1.5, 0.2
            resumed = ProofRunner(agent)
            resumed.resume(result['id'])
            self.assertEqual((agent.seed, agent.temperature, agent.top_p), (42, 0.4, 0.9))
            self.assertEqual(client.requests[1]['options']['seed'], 43)
            self.assertGreaterEqual(resumed.state['tokens_charged'], charged)
            client.backend = 'ollama'
            count = len(client.requests)
            with self.assertRaisesRegex(ValueError, 'original --backend'):
                ProofRunner(agent).resume(result['id'])
            self.assertEqual(len(client.requests), count)

    def test_cli_openai_defaults_and_sampling_reach_agent(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = Mock()
            client.models.return_value = ['square-qwen']
            with patch('sys.argv', ['square-harness', '--backend', 'openai',
                        '--workspace', temporary, '--mode', 'critic', '--seed', '42',
                        '--temperature', '0.4', '--prompt', 'Check x=x.']), \
                    patch.object(cli, 'UI'), patch.object(cli, 'OpenAICompatible', return_value=client) as factory, \
                    patch.object(cli.Agent, 'run', autospec=True, return_value='Checked.') as run:
                self.assertEqual(cli.main(), 0)
                factory.assert_called_once_with('http://localhost:8000', timeout=600)
                agent = run.call_args.args[0]
                self.assertEqual((agent.model, agent.seed, agent.temperature), ('square-qwen', 42, 0.4))

    def test_sampling_validation_precedes_model_network(self):
        with patch.object(cli, 'OpenAICompatible') as factory:
            for option in (['--seed', '-1'], ['--temperature', 'nan'], ['--top-p', 'inf']):
                with self.subTest(option=option), patch('sys.argv', ['square-harness', '--backend', 'openai', *option]), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli.main()
            factory.assert_not_called()

    def test_parallel_cli_does_not_advertise_a_missing_selected_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            client, ui = Mock(), Mock()
            client.models.return_value = ['square-qwen']
            with patch('sys.argv', ['square-harness', '--backend', 'openai',
                        '--workspace', temporary, '--ctx', '32768', '--proof-workers', '3',
                        '--prompt', 'Prove that x=x.']), \
                    patch.object(cli, 'UI', return_value=ui), \
                    patch.object(cli, 'OpenAICompatible', return_value=client), \
                    patch('mathagent.portfolio.run_proof_portfolio', return_value={
                        'status': 'no_complete_candidate', 'answer_path': None}) as run:
                self.assertEqual(cli.main(), 0)
                self.assertEqual(run.call_args.kwargs['workers'], 3)
                output = '\n'.join(str(call.args[0]) for call in ui.say.call_args_list)
                self.assertIn('No candidate was selected', output)
                self.assertNotIn('Selected answer:', output)


if __name__ == '__main__':
    unittest.main()
