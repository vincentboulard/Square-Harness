"""Exact counterexample checker: exact, interval, rejections."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

MISSING = importlib.util.find_spec('sympy') is None or importlib.util.find_spec('mpmath') is None
if not MISSING:
    from mathagent.certify import check

POINCARE = {'version': 1, 'variables': ['t'], 'symbols': {'c': '10'},
            'functions': {'u': {'args': ['x'], 'expr': 'sin(pi*x)'}},
            'assumptions': [{'lhs': 'u(0)', 'relation': '==', 'rhs': '0'}, {'lhs': 'u(1)', 'relation': '==', 'rhs': '0'}],
            'claim': {'lhs': 'integrate(diff(u(t),t)^2,(t,0,1))', 'relation': '>=', 'rhs': 'c*integrate(u(t)^2,(t,0,1))'}}


@unittest.skipIf(MISSING, 'sympy/mpmath missing')
class CertifyTests(unittest.TestCase):
    def cert(self, **changes):
        value = json.loads(json.dumps(POINCARE))
        value.update(changes)
        return value

    def test_interval_certified_counterexample(self):
        report = check(POINCARE)
        self.assertTrue(report['certified'], report)
        self.assertIn('interval', report['checks'][-1]['method'])
        self.assertEqual(report['checks'][0]['method'], 'exact rational arithmetic')

    def test_exact_rational_and_equality_is_not_counterexample(self):
        cert = self.cert(functions={'u': {'args': ['x'], 'expr': 'x*(1-x)'}})
        report = check(cert)
        self.assertFalse(report['certified'])
        self.assertIn('holds at the witness', report['reason'])
        cert['symbols'] = {'c': '11'}
        self.assertTrue(check(cert)['certified'])

    def test_failing_hypothesis_is_not_counterexample(self):
        cert = self.cert(functions={'u': {'args': ['x'], 'expr': 'x'}})
        report = check(cert)
        self.assertFalse(report['certified'])
        self.assertIn('hypothesis fails', report['reason'])

    def test_needs_high_precision(self):
        report = check({'version': 1, 'claim': {'lhs': 'exp(pi*sqrt(163))', 'relation': '==', 'rhs': '262537412640768744'}})
        self.assertTrue(report['certified'])
        self.assertIn('128 bits', report['checks'][0]['method'])

    def test_rejections(self):
        for bad in ('0.5', '__import__(os)', 'x.y', 'lambda: 1', "'a'", 'open(1)'):
            report = check(self.cert(symbols={'c': bad}))
            self.assertFalse(report['certified'], bad)
            self.assertTrue(report['reason'].startswith('Not certified'), bad)
        self.assertFalse(check({'version': 2})['certified'])
        self.assertFalse(check(self.cert(extra=1))['certified'])
        self.assertFalse(check({'version': 1, 'variables': ['x'], 'claim': {'lhs': 'x', 'relation': '<', 'rhs': '0'}})['certified'])

    def test_standalone_script(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'certificate.json'
            path.write_text(json.dumps(POINCARE))
            script = Path(__file__).resolve().parent.parent / 'mathagent' / 'certify.py'
            done = subprocess.run([sys.executable, '-I', str(script), str(path)], capture_output=True, text=True, timeout=120)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertTrue(json.loads(done.stdout)['certified'])


if __name__ == '__main__':
    unittest.main()
