"""Proof presentation never rewrites the candidate that received model review."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mathagent import cli
from mathagent.agent import Agent
from mathagent.ledger import LedgerError, ProofStore
from mathagent.proof import ProofRunner
from mathagent.benchmark import _sequential
from mathagent.portfolio import _branch_result
from mathagent.tools import Workspace


PROOF = ('  # Proof\n\nLet x be an arbitrary real number. Reflexivity gives x=x.\n\n'
         'Scope: this argument makes no assertion about a stronger statement.\n')
CRITIQUE = {'valid_steps': ['Reflexivity holds for every real x.'],
            'first_invalid_step': '', 'reason': '', 'missing_work': '',
            'complete_candidate': True}
CLAIM = {'critical_claim': 'Every real x satisfies x=x.', 'assumptions': ['x is real'],
         'dependencies': [], 'argument': 'Equality is reflexive.', 'disposition': 'supported',
         'objection': '', 'evidence': '', 'next_task': 'Audit the complete proof.',
         'new_progress': True, 'complete_candidate': False, 'resolves': [], 'resolution': ''}
REVIEW = {'claims': [CLAIM], 'complete_candidate': True,
          'next_task': 'Audit the complete proof.', 'strategy_summary': 'Use reflexivity.'}
AUDIT = {'verdict': 'complete', 'explanation': 'Reflexivity proves the stated claim.',
         'objection': '', 'next_task': ''}


class Client:
    host = 'http://offline.invalid'
    timeout = 60

    def __init__(self, replies=()):
        self.replies = iter(replies)
        self.requests = []

    def stream(self, payload):
        self.requests.append(copy.deepcopy(payload))
        try:
            value = next(self.replies)
        except StopIteration:
            raise AssertionError('Presentation must not request additional model inference')
        text = value if isinstance(value, str) else json.dumps(value)
        yield {'message': {'content': text}, 'done': True, 'done_reason': 'stop', 'eval_count': 13}


class ProofExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def runner(self, client):
        return ProofRunner(Agent(client, Workspace(self.root), ctx=16384, predict=512))

    def complete(self):
        client = Client([PROOF, CRITIQUE, REVIEW, AUDIT])
        runner = self.runner(client)
        result = runner.start('Prove x=x for every real x.', max_rounds=1,
                              max_tokens=10000, max_seconds=60)
        return runner, client, result

    def test_export_is_verbatim_audited_candidate_without_extra_inference(self):
        runner, client, result = self.complete()
        path = Path(result['proof_path'])
        candidate = json.loads(runner.store.read_artifact(runner.state['final_audit']['candidate']))
        self.assertEqual(path, runner.store.directory / 'proof.md')
        self.assertEqual(path.read_bytes(), candidate['text'].encode('utf-8'))
        self.assertEqual(path.read_text(encoding='utf-8'), PROOF)
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual(len(client.requests), 4)
        self.assertEqual(runner.state['tokens_charged'], 52)
        self.assertEqual([call['role'] for call in runner.state['calls']],
                         ['solver', 'critic', 'recorder', 'auditor'])
        self.assertIn(PROOF, result['report'])
        self.assertIn('Model reviews are fallible', result['report'])
        self.assertTrue((runner.store.directory / 'ledger.md').is_file())

    def test_partial_candidate_has_diagnostics_but_no_approved_export(self):
        critique = {**CRITIQUE, 'missing_work': 'The remaining case has not been proved.',
                    'complete_candidate': False}
        claim = {**CLAIM, 'disposition': 'gap', 'objection': critique['missing_work']}
        review = {**REVIEW, 'claims': [claim], 'complete_candidate': False,
                  'next_task': critique['missing_work']}
        runner = self.runner(Client(['PARTIAL: one case remains.', critique, review]))
        result = runner.start('Prove the claim in every case.', max_rounds=1, max_seconds=60)
        self.assertIsNone(result['proof_path'])
        self.assertFalse((runner.store.directory / 'proof.md').exists())
        self.assertIn(critique['missing_work'], result['report'])
        self.assertNotEqual(result['status'], 'candidate_complete')

    def test_rejected_whole_proof_audit_never_exports_approval(self):
        audit = {**AUDIT, 'verdict': 'gap', 'explanation': 'The endpoint is omitted.',
                 'objection': 'The endpoint remains unproved.', 'next_task': 'Prove the endpoint.'}
        runner = self.runner(Client([PROOF, CRITIQUE, REVIEW, audit]))
        result = runner.start('Prove the claim including the endpoint.', max_rounds=1,
                              max_seconds=60)
        self.assertIsNone(result['proof_path'])
        self.assertFalse((runner.store.directory / 'proof.md').exists())
        self.assertIn(audit['objection'], result['report'])

    def test_terminal_legacy_resume_recreates_export_without_changing_budget_or_candidate(self):
        runner, _, result = self.complete()
        Path(result['proof_path']).unlink()  # A saved job from before proof.md existed.
        before = copy.deepcopy(runner.state)
        offline = Client()
        resumed = self.runner(offline).resume(result['id'])
        self.assertEqual(Path(resumed['proof_path']).read_text(encoding='utf-8'), PROOF)
        self.assertEqual(offline.requests, [])
        self.assertEqual(ProofStore.load(self.root, result['id']).state, before)

    def test_benchmark_answer_remains_exactly_the_audited_export(self):
        (self.root / 'statement.txt').write_text('Prove x=x for every real x.', encoding='utf-8')
        output = self.root / 'benchmark-job'
        output.mkdir()
        client = Client([PROOF, CRITIQUE, REVIEW, AUDIT])
        agent = Agent(client, Workspace(self.root), ctx=16384, predict=512)
        args = SimpleNamespace(rounds=1, tokens=10000, seconds=60, max_predict=2048)
        outcome = _sequential(agent, 'Prove statement.txt.', output, args)
        self.assertEqual(outcome['status'], 'candidate_complete')
        answer = Path(outcome['answer_path']).read_bytes()
        exported = (Path(outcome['proof_directory']) / 'proof.md').read_bytes()
        self.assertEqual(answer, PROOF.encode('utf-8'))
        self.assertEqual(answer, exported)
        self.assertEqual(outcome['tokens_charged'], 52)
        self.assertEqual(len(client.requests), 4)

    def test_portfolio_selection_receives_the_same_candidate_as_clean_export(self):
        runner, client, result = self.complete()
        branch = self.root / 'branch'
        branch.mkdir()
        (branch / 'worker-result.json').write_text(json.dumps(result), encoding='utf-8')
        outcome = _branch_result({'id': 'branch-000', 'directory': str(branch),
                                  'workspace': str(self.root), 'seed': 0, 'token_budget': 10000})
        self.assertEqual(outcome['candidate']['text'], PROOF)
        self.assertEqual(outcome['candidate']['text'], Path(result['proof_path']).read_text(encoding='utf-8'))
        self.assertEqual(outcome['tokens_charged'], runner.state['tokens_charged'])
        self.assertEqual(len(client.requests), 4)

    def test_proof_export_symlink_is_rejected_without_touching_target(self):
        runner, _, result = self.complete()
        path = Path(result['proof_path'])
        path.unlink()
        target = self.root / 'external.md'
        target.write_text('Preserve this user document.', encoding='utf-8')
        path.symlink_to(target)
        with self.assertRaises(LedgerError):
            runner.store.write_proof(PROOF)
        with self.assertRaises(LedgerError):
            ProofStore.load(self.root, result['id'])
        self.assertEqual(target.read_text(encoding='utf-8'), 'Preserve this user document.')


class ProofExportCliTests(unittest.TestCase):
    def test_success_prints_exact_proof_and_paths_while_incomplete_keeps_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proof = root / 'proof.md'
            proof.write_text(PROOF, encoding='utf-8')
            for status in ('candidate_complete', 'budget_exhausted'):
                with self.subTest(status=status):
                    complete = status == 'candidate_complete'
                    result = {'id': 'saved-proof', 'status': status, 'directory': str(root),
                              'report': 'AUDIT_DIAGNOSTICS: the endpoint remains unresolved.',
                              'proof_path': str(proof) if complete else None,
                              'proof': PROOF if complete else None}
                    ui, client, runner = Mock(), Mock(), Mock()
                    client.models.return_value = ['qwen3.8:27b']
                    runner.start.return_value = result
                    with patch('sys.argv', ['mathagent', '--workspace', str(root), '--prompt', 'Prove it.']), \
                            patch.object(cli, 'UI', return_value=ui), \
                            patch.object(cli, 'Ollama', return_value=client), \
                            patch.object(cli, 'ProofRunner', return_value=runner):
                        self.assertEqual(cli.main(), 0)
                    output = '\n'.join(str(call.args[0]) for call in ui.say.call_args_list)
                    self.assertIn(str(root / 'report.md'), output)
                    if complete:
                        self.assertIn(PROOF, output)
                        self.assertIn(str(proof), output)
                        self.assertNotIn('AUDIT_DIAGNOSTICS', output)
                    else:
                        self.assertIn('AUDIT_DIAGNOSTICS', output)


if __name__ == '__main__':
    unittest.main()
