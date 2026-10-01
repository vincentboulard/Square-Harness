"""Proof mode with optional numerical experiments (refute first, test objections)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from mathagent.agent import Agent
from mathagent.proof import ProofRunner, _experiment_settings
from mathagent.proof_policy import initial_request
from mathagent.tools import Workspace

from test_experiment import CERTIFICATE, FAITHFUL, GOOD_CODE, INTERPRET_REFUTES, PROTOCOL
from test_proof_v05 import Client, approval, event, objection

MISSING = [n for n in ('numpy', 'scipy', 'sympy', 'mpmath', 'matplotlib') if importlib.util.find_spec(n) is None]
SUPPORTS = json.dumps({'verdict': 'supports', 'explanation': 'lambda_1 ~ 9.87 >= 9.', 'key_numbers': ['lambda1 = 9.8696'],
                       'limitations': 'Finite differences.', 'explicit_counterexample': False})
GOAL = 'Prove that int u\'^2 >= 10 int u^2 for every u in H^1_0(0,1).'


@unittest.skipIf(MISSING, 'numerical extras missing')
class ProofExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def start(self, replies, **experiments):
        self.client = Client(replies)
        agent = Agent(self.client, Workspace(self.root, approve=lambda preview: True), ctx=40960, predict=4096,
                      seed=17, temperature=.6, top_p=.95, think=True)
        self.runner = ProofRunner(agent)
        config = dict(permission='ask', run_seconds=60)
        config.update(experiments)
        return self.runner.start(GOAL, max_predict=512, verify_tokens=256, max_tokens=5000, max_rounds=3,
                                 experiments=config)

    def test_settings_validation(self):
        self.assertIsNone(_experiment_settings(None))
        self.assertIsNone(_experiment_settings({'permission': 'ask'}))
        with self.assertRaises(ValueError):
            _experiment_settings({'refute_first': True, 'bogus': 1})
        with self.assertRaises(ValueError):
            _experiment_settings({'refute_first': True, 'permission': 'always'})

    def test_certified_counterexample_refutes_before_solving(self):
        replies = [[event(r)] for r in (PROTOCOL, GOOD_CODE, INTERPRET_REFUTES, CERTIFICATE, FAITHFUL)]
        result = self.start(replies, refute_first=True)
        self.assertEqual(result['status'], 'refuted')
        self.assertEqual(self.runner.state['calls'], [])
        self.assertIn('Numerical experiments', result['report'])
        self.assertEqual(self.runner.state['experiments'][0]['status'], 'certified_counterexample')

    def test_consistent_evidence_reaches_verifier_but_not_initial_solve(self):
        replies = [[event(r)] for r in (PROTOCOL, GOOD_CODE, SUPPORTS)]
        replies += [[event('A proof.')], [event(approval())]]
        result = self.start(replies, refute_first=True)
        self.assertEqual(result['status'], 'candidate_complete')
        solve, review = self.client.requests[3], self.client.requests[4]
        self.assertEqual(solve['messages'], initial_request(GOAL))
        self.assertIn('NUMERICAL CONTEXT FOR THE STATEMENT', review['messages'][0]['content'])

    def test_objection_is_tested_before_repair(self):
        replies = [[event('First proof.')], [event(objection())]]
        replies += [[event(r)] for r in (PROTOCOL, GOOD_CODE, SUPPORTS)]
        replies += [[event('Repaired proof.')], [event(approval())]]
        result = self.start(replies, test_objections=True)
        self.assertEqual(result['status'], 'candidate_complete')
        repair = self.client.requests[5]['messages'][0]['content']
        self.assertIn('NUMERICAL TEST OF THE OBJECTION', repair)
        self.assertEqual(self.runner.state['experiments'][0]['key'], 'objection-V1')

    def test_experiments_off_do_not_inform_the_proof(self):
        replies = [[event(r)] for r in (PROTOCOL, GOOD_CODE)]
        replies += [[event('A proof.')], [event(approval())]]
        result = self.start(replies, refute_first=True, permission='off')
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual(self.runner.state['experiments'][0]['status'], 'not_run')
        self.assertNotIn('NUMERICAL', self.client.requests[3]['messages'][0]['content'])

    def test_interrupted_experiment_pauses_and_resumes_the_proof(self):
        replies = [[event(PROTOCOL)], [KeyboardInterrupt()]]
        result = self.start(replies, refute_first=True)
        self.assertEqual(result['status'], 'paused')
        record = self.runner.state['experiments'][0]
        self.assertEqual(record['status'], 'paused')
        self.assertIsNotNone(record['id'])
        client = Client([[event(r)] for r in (GOOD_CODE, SUPPORTS)] + [[event('A proof.')], [event(approval())]])
        agent = Agent(client, Workspace(self.root, approve=lambda preview: True), ctx=40960, predict=4096,
                      seed=17, temperature=.6, top_p=.95, think=True)
        runner = ProofRunner(agent)
        result = runner.resume(result['id'])
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual(runner.state['experiments'][0]['status'], 'consistent')
        self.assertEqual(runner.state['experiments'][0]['id'], record['id'])


if __name__ == '__main__':
    unittest.main()
