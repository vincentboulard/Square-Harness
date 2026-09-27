"""v0.5 HyperQwen profile invariants and reasoning budgets; no GPU required."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mathagent import benchmark
from mathagent.agent import AgentError
from mathagent.backends import OpenAICompatible, LlamaCpp
from test_benchmark import FakeClient

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('hyper_profile', ROOT / 'scripts/run-hyperqwen-benchmark.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)
ENV = {'SQUARE_OPENAI_THINKING_FRACTION': '0.75',
       'SQUARE_OPENAI_STRUCTURED_THINKING_FRACTION': '0.75'}


class HyperQwenProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, ENV).start()
        self.manifest = benchmark.init_smoke(self.root / 'dataset')

    def args(self):
        return benchmark.parser().parse_args(['--manifest', str(self.manifest),
            '--output', str(self.root / 'output'), '--backend', 'openai',
            '--arms', 'raw-single', 'proof', '--ctx', '40960',
            '--predict', '32768', '--verify-tokens', '16384',
            '--tokens', '120000', '--raw-seconds', '600', '--seconds', '1800'])

    def test_initial_solver_allowance_matches_direct(self):
        args = self.args()
        data, plan = benchmark.preflight(args)
        self.assertEqual(plan['generated_token_ceiling'], 2 * (32768 + 120000))
        self.assertEqual(plan['settings']['predict'], 32768)
        self.assertEqual(plan['openai_thinking_budget_policy'], {'ordinary': .75, 'structured': .75})
        client = FakeClient()
        with patch.object(benchmark, 'create_client', return_value=client):
            result = benchmark.run_job((data['problems'][0], 'raw-single', 0, 42),
                                       self.root / 'output', args, None)
        self.assertEqual(result['budget'], 32768)
        self.assertEqual(result['max_seconds'], 600)
        self.assertEqual(client.requests[0]['options']['num_predict'], 32768)
        call = json.loads(next((self.root / 'output').rglob('call.json')).read_text())
        self.assertEqual(call['max_seconds'], 600)

    def test_raw_output_and_time_invalid_values_fail_preflight(self):
        for key, value in [('predict', 40960), ('predict', 127),
                           ('raw_seconds', 0), ('raw_seconds', float('nan'))]:
            args = self.args()
            setattr(args, key, value)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                benchmark.preflight(args)

    def test_native_budget_leaves_answer_space_without_disabling_thinking(self):
        client = OpenAICompatible()
        native = {'model': 'test', 'messages': [{'role': 'user', 'content': 'A problem'}],
                  'think': True, 'options': {'num_predict': 32768}}
        wire = client._payload(native)
        self.assertEqual(wire['thinking_token_budget'], 24576)
        self.assertEqual(wire['max_completion_tokens'], 32768)
        self.assertTrue(wire['chat_template_kwargs']['enable_thinking'])
        native.update(format={'type': 'object'})
        native['options']['num_predict'] = 16384
        self.assertEqual(client._payload(native)['thinking_token_budget'], 12288)
        native['think'] = False
        self.assertNotIn('thinking_token_budget', client._payload(native))
        native['think'] = True
        self.assertNotIn('thinking_token_budget', LlamaCpp()._payload(native))

    def test_policy_is_opt_in_and_rejects_invalid_fractions(self):
        for value in ('-1', '1', 'nan', 'garbage'):
            with patch.dict(os.environ, {'SQUARE_OPENAI_THINKING_FRACTION': value}), self.assertRaises(AgentError):
                OpenAICompatible()
        with patch.dict(os.environ, {}, clear=True):
            client = OpenAICompatible()
            self.assertEqual(client.thinking_budget_policy, {})

    def test_changed_policy_after_preflight_stops_before_output_or_network(self):
        args = self.args()
        data, plan = benchmark.preflight(args)
        with patch.dict(os.environ, {'SQUARE_OPENAI_THINKING_FRACTION': '.25'}), \
                patch.object(benchmark, 'create_client', side_effect=AssertionError('network')):
            with self.assertRaisesRegex(ValueError, 'policy changed'):
                benchmark.execute(args, data, plan)
        self.assertFalse(args.output.exists())

    def test_frozen_wrapper_and_acceptance_reject_source_drift(self):
        items = []
        for i in range(10):
            name = f'p{i}.md'
            (self.manifest.parent / name).write_text('Prove x=x for real x.')
            items.append({'id': f'P{i}', 'statement': name})
        self.manifest.write_text(json.dumps({'version': 1, 'name': 'fixture', 'problems': items}))
        lock = json.loads((ROOT / 'deployment/hyperqwen-a10.lock.json').read_text())
        launch_path = self.root / 'launch.json'
        launch_path.write_text(json.dumps({'image': lock['image'], 'env': lock['server_env']}))
        args = argparse.Namespace(manifest=self.manifest, output=self.root/'output',
                                  launch_record=launch_path, acceptance=None, dry_run=True,
                                  arms=['raw-single', 'proof'])
        with patch.object(profile, 'verify_live', side_effect=AssertionError('network')):
            _, _, _, _, plan = profile.prepare(args)
        self.assertEqual(plan['jobs'], 20)
        self.assertEqual(plan['generated_token_ceiling'], 10 * 32768 + 10 * 80000)
        self.assertEqual(plan['nominal_job_time_ceiling_seconds'], 36000)
        proof_only = argparse.Namespace(**{**vars(args), 'arms': ['proof'], 'output': self.root / 'proof-only'})
        _, _, parsed, _, solo = profile.prepare(proof_only)
        self.assertEqual((solo['jobs'], solo['generated_token_ceiling']), (10, 800000))
        self.assertEqual(solo['nominal_job_time_ceiling_seconds'], 18000)
        self.assertEqual((parsed.repair_tokens, parsed.min_solve_tokens, parsed.rounds), (14000, 14000, 2))
        accepted = {'ok': True, 'launch_sha256': profile.read(launch_path)[1],
                    'profile_lock_sha256': profile.read(ROOT / 'deployment/hyperqwen-a10.lock.json')[1],
                    'code_sha256': plan['code_sha256'],
                    'thinking_budget_policy': plan['openai_thinking_budget_policy']}
        args.acceptance = self.root / 'acceptance.json'
        args.acceptance.write_text(json.dumps(accepted))
        profile.prepare(args)
        accepted['code_sha256'] = {}
        args.acceptance.write_text(json.dumps(accepted))
        with self.assertRaisesRegex(ValueError, 'Acceptance'):
            profile.prepare(args)


if __name__ == '__main__':
    unittest.main()
