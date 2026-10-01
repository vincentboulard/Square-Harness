"""Experiment sandbox: limits, network, figures, permissions."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from mathagent import sandbox


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def run_code(self, code, name='run', **limits):
        limits.setdefault('seconds', 20)
        return sandbox.run_python(code, self.base / name, cache=self.base / 'cache', **limits)

    def test_output_results_and_exit_code(self):
        result = self.run_code('import json\nprint("hello")\njson.dump({"a": 1}, open("results.json", "w"))\nraise SystemExit(3)')
        self.assertEqual(result['returncode'], 3)
        self.assertFalse(result['ok'])
        self.assertEqual(result['stdout'], 'hello\n')
        self.assertEqual(result['results'], {'a': 1})
        self.assertEqual(result['isolation'], sandbox.available_isolation())

    def test_exception_is_reported(self):
        result = self.run_code('raise ValueError("boom")')
        self.assertEqual(result['returncode'], 1)
        self.assertIn('ValueError: boom', result['stderr'])

    def test_cpu_limit(self):
        result = self.run_code('while True:\n    pass', seconds=2)
        self.assertFalse(result['ok'])
        self.assertIsNotNone(result['limit_reason'])
        self.assertLess(result['seconds'], 15)

    @unittest.skipUnless(sandbox.sys.platform.startswith('linux'), 'RLIMIT_AS is enforced on Linux')
    def test_memory_limit(self):
        result = self.run_code('x = bytearray(2 * 1024 ** 3)', memory_mb=512)
        self.assertFalse(result['ok'])
        self.assertIn('memory', result['limit_reason'])

    def test_python_level_network_guard(self):
        result = self.run_code('import socket\ntry:\n    socket.create_connection(("1.1.1.1", 80), timeout=2)\n'
                               'except OSError as e:\n    print("blocked:", e)')
        self.assertIn('blocked', result['stdout'])

    def test_clean_environment(self):
        result = self.run_code('import os\nprint(sorted(k for k in os.environ if "TOKEN" in k or "KEY" in k))')
        self.assertEqual(result['stdout'].strip(), '[]')

    @unittest.skipIf(importlib.util.find_spec('matplotlib') is None, 'matplotlib missing')
    def test_open_figures_are_saved_by_label(self):
        result = self.run_code('import matplotlib.pyplot as plt\nplt.figure("my plot!")\nplt.plot([0, 1])', seconds=60)
        self.assertTrue(result['ok'], result['stderr'])
        self.assertEqual(result['figures'], ['my-plot.png', 'my-plot.svg'])

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            sandbox.run_python('', self.base / 'empty')
        with self.assertRaises(ValueError):
            self.run_code('pass', name='bad', seconds=0)

    def test_permissions(self):
        asked = []

        def approve(preview):
            asked.append(preview)
            return True
        self.assertEqual(sandbox.permitted('off', approve, 'code'), (False, sandbox.permitted('off', approve, '')[1]))
        self.assertEqual(asked, [])
        allowed, _ = sandbox.permitted('ask', approve, 'print(1)')
        self.assertTrue(allowed)
        self.assertTrue(asked[0].startswith('RUN NUMERICAL EXPERIMENT'))
        self.assertTrue(asked[0].endswith('print(1)'))
        allowed, how = sandbox.permitted('auto', lambda preview: False, 'x')
        self.assertEqual(allowed, sandbox.available_isolation() in sandbox.ISOLATED)
        with self.assertRaises(ValueError):
            sandbox.permitted('always', approve, 'x')


if __name__ == '__main__':
    unittest.main()
