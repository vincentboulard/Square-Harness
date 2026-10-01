"""Experiment workflow: protocol, sandboxed runs, controller-decided status, certificates."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from mathagent.agent import Agent
from mathagent.experiment import ExperimentRunner, extract_code, extract_json
from mathagent.research import ResearchRunner
from mathagent.tools import Workspace

NEEDS = ('numpy', 'scipy', 'sympy', 'mpmath', 'matplotlib')
MISSING = [name for name in NEEDS if importlib.util.find_spec(name) is None]


def reply(text, count=50):
    return [{'message': {'content': text}, 'done': True, 'done_reason': 'stop',
             'eval_count': count, 'prompt_eval_count': 200}]


class FakeClient:
    host = 'http://fake.invalid'
    timeout = 60

    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def stream(self, payload):
        self.requests.append(copy.deepcopy(payload))
        yield from next(self.replies)


PROTOCOL = json.dumps({
    'hypothesis': 'For all u in H^1_0(0,1), int u\'^2 >= 10 int u^2.',
    'quantity': 'First Dirichlet eigenvalue of -u\'\' on (0,1).',
    'support_criterion': 'lambda_1 >= 10 after convergence.',
    'refutation_criterion': 'lambda_1 < 10 after convergence.',
    'validation_case': 'Interval of length 1/2: lambda_1 = 4 pi^2.',
    'discretisation_control': 'n = 50, 100, 200, 400.',
    'search_family': 'Eigenfunctions only.', 'figures': 'Convergence plot.'})

GOOD_CODE = '''```python
from mathagent import numerics as sq
import numpy as np
sq.validate('half interval', sq.smallest_eigenvalues(sq.laplacian_1d(400, 0.5), 1)[0], 4*np.pi**2, rtol=1e-3)
ns = [50, 100, 200, 400]
vals = [sq.smallest_eigenvalues(sq.laplacian_1d(n), 1)[0] for n in ns]
c = sq.convergence('lambda1', ns, vals)
sq.record('lambda1', c['extrapolated'])
sq.plot_curves('lambda1', ns, {'lambda1': vals})
print('lambda1 =', vals[-1])
```'''

INTERPRET_REFUTES = json.dumps({'verdict': 'refutes', 'explanation': 'lambda_1 ~ 9.87 < 10.',
                                'key_numbers': ['lambda1 = 9.8696'], 'limitations': 'Finite differences.',
                                'explicit_counterexample': True})
CERTIFICATE = json.dumps({'version': 1, 'statement': 'int u\'^2 >= 10 int u^2 on H^1_0(0,1)', 'variables': ['t'],
                          'functions': {'u': {'args': ['x'], 'expr': 'sin(pi*x)'}},
                          'assumptions': [{'lhs': 'u(0)', 'relation': '==', 'rhs': '0'},
                                          {'lhs': 'u(1)', 'relation': '==', 'rhs': '0'}],
                          'claim': {'lhs': 'integrate(diff(u(t),t)^2,(t,0,1))', 'relation': '>=',
                                    'rhs': '10*integrate(u(t)^2,(t,0,1))'}})
FAITHFUL = json.dumps({'verdict': 'faithful', 'explanation': 'Same hypotheses and conclusion.'})


@unittest.skipIf(MISSING, 'numerical extras missing: ' + ', '.join(MISSING))
class ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.prompts = []

    def tearDown(self):
        self.tmp.cleanup()

    def runner(self, replies, approve=True):
        self.client = FakeClient([reply(r) for r in replies])

        def ask(preview):
            self.prompts.append(preview)
            return approve
        return ExperimentRunner(Agent(self.client, Workspace(self.root, approve=ask), ctx=32768, predict=4096))

    def test_certified_counterexample_end_to_end(self):
        result = self.runner([PROTOCOL, GOOD_CODE, INTERPRET_REFUTES, CERTIFICATE, FAITHFUL]).start(
            'Prove that int u\'^2 >= 10 int u^2 for u in H^1_0(0,1).', permission='ask', run_seconds=60)
        self.assertEqual(result['status'], 'certified_counterexample', result['report'][:3000])
        directory = Path(result['directory'])
        state = json.loads((directory / 'state.json').read_text())
        self.assertTrue(state['certificate_check']['certified'])
        self.assertTrue((directory / 'certificate' / 'verify_certificate.py').is_file())
        self.assertTrue((directory / 'runs' / 'run-1' / 'figures' / 'lambda1.png').is_file())
        self.assertIn('RUN NUMERICAL EXPERIMENT', self.prompts[0])
        self.assertIn('Certified counterexample', result['report'])
        self.assertEqual(ResearchRunner.inspect(self.root, state['id'])['kind'], 'experiment')

    def test_unfaithful_certificate_is_only_numerical_evidence(self):
        unfaithful = json.dumps({'verdict': 'not_faithful', 'explanation': 'Boundary condition dropped.'})
        result = self.runner([PROTOCOL, GOOD_CODE, INTERPRET_REFUTES, CERTIFICATE, unfaithful]).start(
            'Claim', permission='ask', run_seconds=60)
        self.assertEqual(result['status'], 'evidence_against')
        self.assertIn('did not confirm', result['report'])

    def test_wrong_certificate_is_not_certified_after_retry(self):
        wrong = json.loads(CERTIFICATE)
        wrong['claim']['rhs'] = '9*integrate(u(t)^2,(t,0,1))'
        result = self.runner([PROTOCOL, GOOD_CODE, INTERPRET_REFUTES, json.dumps(wrong), json.dumps(wrong)]).start(
            'Claim', permission='ask', run_seconds=60)
        self.assertEqual(result['status'], 'evidence_against')
        state = json.loads((Path(result['directory']) / 'state.json').read_text())
        self.assertEqual(state['certificate_attempts'], 2)
        self.assertFalse(state['certificate_check']['certified'])

    def test_permission_off_saves_code_without_running(self):
        result = self.runner([PROTOCOL, GOOD_CODE]).start('Claim', permission='off')
        self.assertEqual(result['status'], 'not_run')
        self.assertEqual(self.prompts, [])
        self.assertIn('sq.validate', result['report'])
        self.assertFalse((Path(result['directory']) / 'runs').exists())

    def test_declined_run(self):
        result = self.runner([PROTOCOL, GOOD_CODE], approve=False).start('Claim', permission='ask')
        self.assertEqual(result['status'], 'not_run')
        self.assertEqual(len(self.prompts), 1)

    def test_failing_code_is_fixed_then_consistent(self):
        broken = '```python\nraise RuntimeError("boom")\n```'
        supports = json.dumps({'verdict': 'supports', 'explanation': 'ok', 'key_numbers': [], 'limitations': '',
                               'explicit_counterexample': False})
        result = self.runner([PROTOCOL, broken, GOOD_CODE, supports]).start('Claim', permission='ask', run_seconds=60)
        self.assertEqual(result['status'], 'consistent')
        fix_prompt = self.client.requests[2]['messages'][1]['content']
        self.assertIn('boom', fix_prompt)

    def test_missing_validation_is_unvalidated(self):
        code = '```python\nfrom mathagent import numerics as sq\nsq.record("x", 1)\n```'
        supports = json.dumps({'verdict': 'supports', 'explanation': 'ok', 'key_numbers': [], 'limitations': '',
                               'explicit_counterexample': False})
        result = self.runner([PROTOCOL, code, code, supports]).start('Claim', permission='ask', run_seconds=60, max_rounds=2)
        self.assertEqual(result['status'], 'unvalidated')

    def test_failed_validation_overrides_model_verdict(self):
        code = ('```python\nfrom mathagent import numerics as sq\nsq.validate("bad", 1.0, 2.0)\n'
                'sq.convergence("q", [1, 2, 4], [1.0, 1.0, 1.0])\n```')
        result = self.runner([PROTOCOL, code, INTERPRET_REFUTES, CERTIFICATE, FAITHFUL]).start(
            'Claim', permission='ask', run_seconds=60, max_rounds=1)
        self.assertNotEqual(result['status'], 'evidence_against')

    def test_syntax_errors_exhaust_runs(self):
        bad = '```python\ndef (:\n```'
        result = self.runner([PROTOCOL, bad, bad, bad]).start('Claim', permission='ask', max_rounds=2)
        self.assertEqual(result['status'], 'run_failed')
        self.assertEqual(self.prompts, [])

    def test_resume_finished_job_needs_no_model(self):
        result = self.runner([PROTOCOL, GOOD_CODE]).start('Claim', permission='off')
        again = ExperimentRunner(Agent(FakeClient([]), Workspace(self.root), ctx=32768, predict=4096)).resume(
            Path(result['directory']).name)
        self.assertEqual(again['status'], 'not_run')


class ExtractTest(unittest.TestCase):
    def test_extract_code(self):
        self.assertEqual(extract_code('text\n```python\nx = 1\n```\nmore')[0], 'x = 1')
        self.assertIsNotNone(extract_code('```python\ndef (\n```')[1])

    def test_extract_json(self):
        self.assertEqual(extract_json('Here:\n```json\n{"a": 1}\n```'), {'a': 1})
        self.assertEqual(extract_json('noise {"a": 2} trailing'), {'a': 2})
        with self.assertRaises(ValueError):
            extract_json('none')


if __name__ == '__main__':
    unittest.main()
