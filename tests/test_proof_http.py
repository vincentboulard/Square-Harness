"""CLI-to-HTTP v0.5 integration: two roles, exact answer, offline inspection."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest


class ProofHttpIntegration(unittest.TestCase):
    def test_solve_verify_stop_and_offline_completed_resume(self):
        requests = []
        model = 'local-test-model'
        proof = 'Let x be an arbitrary real number. Reflexivity gives x=x.\n'
        verdict = {'explanation': 'No invalid inference was found; the proof covers every real x.',
                   'issues': [], 'verdict': 'no_issue_found'}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({'models': [{'name': model}]}).encode())

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(payload)
                self.send_response(200)
                self.end_headers()
                body = proof if len(requests) == 1 else json.dumps(verdict)
                events = [{'message': {'thinking': 'Check the quantifier and reflexivity.'}, 'done': False},
                          {'message': {'content': body}, 'done': True, 'done_reason': 'stop',
                           'eval_count': 50, 'prompt_eval_count': 80}]
                for event in events:
                    self.wfile.write(json.dumps(event).encode() + b'\n')

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'identity.tex').write_text('For every real x, prove x=x.', encoding='utf-8')
            base = [sys.executable, '-m', 'mathagent', '--workspace', str(root),
                    '--host', f'http://127.0.0.1:{server.server_port}']
            try:
                run = subprocess.run(base + ['--model', model, '--ctx', '16384',
                    '--proof-solve-tokens', '2048', '--proof-verify-tokens', '1024',
                    '--proof-tokens', '8192', '--proof-rounds', '2', '--proof-seconds', '30',
                    '--proof-file', 'identity.tex', '--no-think',
                    '--prompt', 'Prove the supplied statement.'], capture_output=True, text=True, timeout=20)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                self.assertIn('candidate_complete', run.stdout)
                self.assertEqual(len(requests), 2, 'No recorder, planner or duplicate audit')
                self.assertTrue(all(request['think'] for request in requests))
                self.assertTrue(all('tools' not in request for request in requests))
                self.assertEqual([m['role'] for m in requests[0]['messages']], ['user'])
                self.assertIn('For every real x', str(requests[0]['messages']))
                self.assertIn(proof, str(requests[1]['messages'][1]['content']) if len(requests[1]['messages']) > 1
                              else requests[1]['messages'][0]['content'])
                folder = next((root / '.mathagent/proofs').iterdir())
                state = json.loads((folder / 'state.json').read_text())
                self.assertEqual(state['tokens_charged'], 100)
                self.assertEqual((folder / 'proof.md').read_text(), proof)
                self.assertEqual(state['version'], 2)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
            for command in ('/proofs', f'/proof-report {state["id"]}', f'/resume {state["id"]}'):
                inspected = subprocess.run(base + ['--prompt', command], capture_output=True, text=True, timeout=10)
                self.assertEqual(inspected.returncode, 0, inspected.stdout + inspected.stderr)
                self.assertIn(state['id'], inspected.stdout)
            self.assertEqual(len(requests), 2)


if __name__ == '__main__':
    unittest.main()
