"""Experiments in the visual interface: start, approval, figures, effort, routing."""
import importlib.util
import json
import unittest

from mathagent import cli
from mathagent.gui import effort
from mathagent.router import JOB_MODES

from test_experiment import GOOD_CODE, PROTOCOL
from test_gui import GuiCase, text

MISSING = [n for n in ('numpy', 'scipy', 'sympy', 'mpmath', 'matplotlib') if importlib.util.find_spec(n) is None]
SUPPORTS = json.dumps({'verdict': 'supports', 'explanation': 'ok', 'key_numbers': [], 'limitations': '',
                       'explicit_counterexample': False})


@unittest.skipIf(MISSING, 'numerical extras missing')
class ExperimentGuiTests(GuiCase):
    def test_status_reports_isolation_and_defaults(self):
        status = self.ok('GET', '/api/status')
        self.assertIn(status['isolation'], ('bwrap', 'seatbelt', 'netns', 'none'))
        self.assertEqual(status['defaults']['experiments'], 'ask')
        self.assertFalse(status['defaults']['proof_refute_first'])

    def test_experiment_runs_after_approval_and_serves_figures(self):
        self.fake.replies = [text(PROTOCOL), text(GOOD_CODE), text(SUPPORTS)]
        started = self.ok('POST', '/api/experiments', {'goal': 'Is lambda_1 >= 9 on (0,1)?', 'permission': 'ask',
                                                       'run_seconds': 60, 'rounds': 2, 'tokens': 16000, 'seconds': 300})
        self.assertEqual(started['task']['label'], 'Experiment')
        approval = self.wait_for(lambda: self.ok('GET', '/api/task')['approvals'], timeout=20, message='an approval')[0]
        self.assertEqual(approval['kind'], 'experiment')
        self.assertIn('sq.validate', approval['preview'])
        self.ok('POST', f'/api/approvals/{approval["id"]}', {'approve': True})
        self.assertEqual(self.wait_task(timeout=60)['state'], 'done')
        detail = self.ok('GET', f'/api/research/{started["id"]}')
        self.assertEqual(detail['kind'], 'experiment')
        self.assertEqual(detail['status'], 'consistent')
        self.assertEqual(detail['runs'][0]['folder'], 'runs/run-1')
        self.assertIn('lambda1.png', detail['runs'][0]['result']['figures'])
        status, body, headers = self.request('GET', f'/api/research/{started["id"]}/figures/run-1/lambda1.png')
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'image/png')
        status, _, headers = self.request('GET', f'/api/research/{started["id"]}/figures/run-1/lambda1.svg')
        self.assertEqual(status, 200)
        self.assertIn('attachment', headers['Content-Disposition'])
        self.assertEqual(self.request('GET', f'/api/research/{started["id"]}/figures/run-1/missing.png')[0], 404)
        self.assertEqual(self.request('GET', f'/api/research/{started["id"]}/figures/run-9/lambda1.png')[0], 404)
        self.assertEqual(self.ok('GET', '/api/research')['jobs'][0]['kind'], 'experiment')

    def test_invalid_permission_is_rejected(self):
        status, body, _ = self.request('POST', '/api/experiments', {'goal': 'x', 'permission': 'always'})
        self.assertEqual(status, 400, body)


class ExperimentRoutingTests(unittest.TestCase):
    def test_effort_limits_and_router_mode(self):
        args = cli.parser().parse_args([])
        limits = effort.limits('experiment', 'low', args)
        self.assertEqual(limits['rounds'], 2)
        self.assertGreaterEqual(limits['seconds'], 300)
        self.assertEqual(set(limits), {'rounds', 'tokens', 'input_tokens', 'seconds'})
        self.assertIn('experiment', JOB_MODES)


if __name__ == '__main__':
    unittest.main()
