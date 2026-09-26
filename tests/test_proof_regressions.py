"""Behavioral regressions for compaction, route selection and proof assembly."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from mathagent.agent import Agent
from mathagent.ledger import ProofStore
from mathagent.proof import ProofRunner
from mathagent.tools import Workspace


def response(value):
    text = value if isinstance(value, str) else json.dumps(value)
    return [{'message': {'content': text}, 'done': True,
             'done_reason': 'stop', 'eval_count': 20}]


CRITIC = {'valid_steps': ['Equality is reflexive for every real x.'],
          'first_invalid_step': '', 'reason': '', 'missing_work': '',
          'complete_candidate': True}
REVIEW = {
    'claims': [{'critical_claim': 'For every real x, x=x.', 'assumptions': [],
                'dependencies': [], 'argument': 'Equality is reflexive.',
                'disposition': 'supported', 'objection': '', 'evidence': '',
                'next_task': 'Audit the written proof.', 'new_progress': True,
                'complete_candidate': False, 'resolves': [], 'resolution': ''}],
    'complete_candidate': True, 'next_task': 'Audit the written proof.',
    'strategy_summary': 'Applied reflexivity of equality.',
}
AUDIT = {'verdict': 'complete', 'explanation': 'Reflexivity applies to every real x.',
         'objection': '', 'next_task': ''}
PLAN = {'approach': 'Inspect the defining relation.',
        'task': 'Use the definition of equality for arbitrary real x.',
        'difference': 'Do not impose strict positivity.',
        'deliverable': 'A derivation covering all real values.'}


class Client:
    host = 'http://regression.invalid'
    timeout = 60

    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def stream(self, payload):
        self.requests.append(copy.deepcopy(payload))
        try:
            yield from next(self.replies)
        except StopIteration:
            raise AssertionError('Unexpected inference call') from None


class ProofRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def runner(self, replies):
        client = Client(replies)
        runner = ProofRunner(Agent(client, Workspace(self.root), model='test-model',
                                   ctx=16384, predict=2048, think=True))
        settings = {'max_rounds': 10, 'max_tokens': 60000, 'max_seconds': 1800,
                    'model': 'test-model', 'ctx': 16384, 'predict': 2048,
                    'max_predict': 8192, 'think': True, 'host': client.host}
        runner.store = ProofStore.create(self.root, 'Prove x=x for every real x.', settings)
        return runner, client

    @staticmethod
    def claims(argument):
        return [{'id': f'C{n}', 'round': n // 4 + 1, 'status': 'reviewed',
                 'statement': f'Intermediate claim {n}', 'assumptions': [],
                 'dependencies': [], 'argument': argument, 'objection': '',
                 'evidence': '', 'resolves': [], 'resolution': '',
                 'candidate_artifact': '0001-candidate.md', 'review_artifact': None,
                 'next_task': ''} for n in range(1, 21)]

    def assert_request_fits(self, request):
        cap = request['options']['num_predict']
        actual = len(json.dumps(request, ensure_ascii=False).encode())
        self.assertLessEqual(actual, (request['options']['num_ctx'] - cap - 256) * 3)

    def test_recorder_compaction_counts_serialized_schema_and_escaping(self):
        # Plain text reproduces the schema-overhead boundary; backslashes and
        # quotes additionally exercise serialization growth in LaTeX-like text.
        for argument in ('a' * 1300, ('\\alpha = "quoted"; ' * 100)):
            with self.subTest(argument_prefix=argument[:30]):
                runner, client = self.runner([response(REVIEW)])
                runner.state['claims'] = self.claims(argument)
                original = copy.deepcopy(runner.state['claims'])
                draft = {'text': 'For every real x, reflexivity gives x=x.',
                         'thinking': '', 'complete': True}
                result = runner._review(draft, copy.deepcopy(CRITIC))
                self.assertTrue(result['complete_candidate'])
                self.assertEqual(len(client.requests), 1)
                request = client.requests[0]
                self.assertIn('format', request)
                self.assert_request_fits(request)
                self.assertIn('Records omitted from working context', request['messages'][1]['content'])
                self.assertLess(len(result['_visible_claims']), len(original))
                self.assertEqual(runner.state['claims'], original)

    def test_solver_compaction_counts_tool_schemas_before_dispatch(self):
        runner, client = self.runner([response('For every real x, reflexivity gives x=x.')])
        runner.state['claims'] = self.claims('Argument with \\symbols and "quotes". ' * 80)
        original = copy.deepcopy(runner.state['claims'])
        runner.state['pending'] = {'index': 1, 'phase': 'solve', 'fresh': False,
                                   'think': True, 'task': 'Write the next argument.'}
        result = runner._solve(runner.state['pending'])
        self.assertTrue(result['complete'])
        self.assertEqual(len(client.requests), 1)
        self.assertIn('tools', client.requests[0])
        self.assert_request_fits(client.requests[0])
        self.assertEqual(runner.state['claims'], original)

    def test_identical_previous_plan_is_rejected_despite_deliverable_suffix(self):
        runner, client = self.runner([response(PLAN)])
        runner.state['rounds'] = [{'index': 1, 'plan': copy.deepcopy(PLAN),
                                  'task': PLAN['task'] + '\nRequired deliverable: ' + PLAN['deliverable'],
                                  'strategy_summary': 'Already attempted without resolving the goal.'}]
        result = runner._plan()
        self.assertNotEqual(result['task'], PLAN['task'])
        self.assertIn('repeated', result['difference'].lower())
        self.assertEqual(len(client.requests), 1)

    def test_ready_assembly_overrides_stagnation_and_fourth_round_exploration(self):
        runner, client = self.runner([response('For every real x, reflexivity gives x=x.'),
                                     response(CRITIC), response(AUDIT)])
        assembly = 'Recheck the checkpoint, then write a self-contained proof for the final audit.'
        runner.state.update(status='paused', rounds_started=3, stagnant_rounds=3,
                            truncation_streak=2, assemble_next=True, next_task=assembly,
                            rounds=[{'index': n, 'task': 'Earlier route.', 'strategy_summary': 'Incomplete.'}
                                    for n in range(1, 4)])
        runner.state['settings']['max_rounds'] = 4
        runner.store.save()
        result = runner.resume(runner.state['id'])
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual([call['role'] for call in runner.state['calls']],
                         ['solver', 'critic', 'auditor'])
        self.assertFalse(client.requests[0]['think'])
        self.assertIn(assembly, client.requests[0]['messages'][1]['content'])
        self.assertEqual(runner.state['rounds_started'], 4)
        self.assertFalse(runner.state['rounds'][-1]['fresh'])

    def test_critic_with_unresolved_reason_cannot_approve_or_request_assembly(self):
        inconsistent = dict(CRITIC, reason='The key inference cannot be justified from the hypotheses.')
        runner, _ = self.runner([response(inconsistent)])
        result = runner._critic({'text': 'An unsupported assertion.', 'thinking': '', 'complete': True})
        self.assertFalse(result['complete_candidate'])
        self.assertFalse(result['ready_for_assembly'])


if __name__ == '__main__':
    unittest.main()
