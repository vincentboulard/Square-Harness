"""Numerical helpers against exact answers."""
import importlib.util
import json
import os
import tempfile
import unittest

MISSING = [n for n in ('numpy', 'scipy', 'matplotlib') if importlib.util.find_spec(n) is None]
if not MISSING:
    import numpy as np
    from mathagent import numerics as sq


@unittest.skipIf(MISSING, 'numerical extras missing')
class NumericsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self.cwd)

    def results(self):
        with open('results.json') as handle:
            return json.load(handle)

    def test_dirichlet_eigenvalues_second_order(self):
        ns = [20, 40, 80, 160]
        values = [sq.smallest_eigenvalues(sq.laplacian_1d(n), 1)[0] for n in ns]
        entry = sq.convergence('lambda1', ns, values, exact=np.pi ** 2)
        self.assertAlmostEqual(entry['observed_order'], 2, delta=0.1)
        self.assertAlmostEqual(entry['extrapolated'], np.pi ** 2, delta=1e-4)
        self.assertTrue(sq.validate('rect', sq.smallest_eigenvalues(sq.laplacian_2d(40), 3),
                                    sq.dirichlet_eigenvalues_rectangle(3), rtol=1e-2))
        self.assertTrue(self.results()['validations'][0]['passed'])

    def test_neumann_and_periodic_spectra(self):
        neumann = sq.smallest_eigenvalues(sq.laplacian_1d(200, bc='neumann'), 2)
        self.assertAlmostEqual(neumann[0], 0, delta=1e-8)
        self.assertAlmostEqual(neumann[1], np.pi ** 2, delta=1e-2)
        periodic = sq.smallest_eigenvalues(sq.laplacian_1d(200, bc='periodic'), 2)
        self.assertAlmostEqual(periodic[1], 4 * np.pi ** 2, delta=1e-1)

    def test_heat_wave_and_gramian(self):
        x = sq.grid(200)
        _, _, U = sq.heat_1d(np.sin(np.pi * x), 0.1, 200)
        self.assertAlmostEqual(U[-1].max(), np.exp(-np.pi ** 2 * 0.1), delta=1e-4)
        _, _, U, E = sq.wave_1d(np.sin(np.pi * x), 0 * x, 2.0, 1000)
        self.assertLess((E.max() - E.min()) / E[0], 1e-3)
        with self.assertRaises(ValueError):
            sq.wave_1d(np.sin(np.pi * x), 0 * x, 2.0, 10)
        _, lam = sq.observability_gramian(np.array([[0, 1], [-1, 0]]), np.array([[1, 0]]), 2 * np.pi)
        self.assertAlmostEqual(lam, np.pi, delta=1e-8)

    def test_failed_validation_and_search(self):
        self.assertFalse(sq.validate('bad', 1.0, 2.0))
        x, value = sq.maximize(lambda p: -(p[0] - 0.3) ** 2, [(0, 1)], name='toy')
        self.assertAlmostEqual(x[0], 0.3, delta=1e-5)
        data = self.results()
        self.assertFalse(data['validations'][0]['passed'])
        self.assertEqual(data['searches'][0]['name'], 'toy')

    def test_record_converts_numpy(self):
        sq.record('array', np.arange(3))
        sq.record('scalar', np.float64(2.5))
        sq.record('nan', float('nan'))
        values = self.results()['values']
        self.assertEqual(values, {'array': [0, 1, 2], 'scalar': 2.5, 'nan': 'nan'})


if __name__ == '__main__':
    unittest.main()
