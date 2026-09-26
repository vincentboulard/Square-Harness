"""Real HTTP/CLI proof workflow against deterministic simulated model responses."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest


class PersistentProofHttpIntegration(unittest.TestCase):
    def test_tool_solver_critic_record_audit_then_offline_inspection_and_resume(self):
        requests, endpoints = [], []
        model = 'local-test-model:latest'
        statement = r'\paragraph{Task.} Prove that every real number $x$ satisfies $x=x$.' + '\n'
        proof = ('Let x be an arbitrary real number. Equality is reflexive, so x=x. '
                 'Since x was arbitrary, the statement holds for every real number. '
                 'This is a complete proof; there are no remaining obligations.')
        review = {
            'critical_claim': 'For every real number x, x=x.',
            'assumptions': ['x is a real number.'], 'dependencies': [],
            'argument': proof, 'disposition': 'supported', 'objection': '',
            'evidence': 'Reflexivity of equality applies to every real number.',
            'next_task': 'Audit the self-contained proof against the original statement.',
            'new_progress': True, 'complete_candidate': False,
            'resolves': [], 'resolution': '',
        }
        critic = {
            'valid_steps': ['For arbitrary real x, reflexivity gives x=x.'],
            'first_invalid_step': '', 'reason': '', 'missing_work': '',
            'complete_candidate': True,
        }
        recorded = {
            'claims': [review], 'complete_candidate': True,
            'next_task': 'Audit the complete proof.',
            'strategy_summary': 'Apply reflexivity to an arbitrary real number.',
        }
        audit = {
            'verdict': 'complete',
            'explanation': 'The proof applies reflexivity to an arbitrary real x and covers the stated quantifier.',
            'objection': '', 'next_task': '',
        }

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                endpoints.append(('GET', self.path))
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'models': [{'name': model}]}).encode())

            def do_POST(self):
                endpoints.append(('POST', self.path))
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(payload)
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson')
                self.end_headers()
                index = len(requests)
                if index == 1:
                    events = [
                        {'message': {'thinking': 'Inspect the original source.'}, 'done': False},
                        {'message': {'tool_calls': [{'function': {'name': 'read_file',
                            'arguments': {'path': 'identity.tex'}}}]},
                         'done': True, 'done_reason': 'stop', 'eval_count': 10},
                    ]
                elif index == 2:
                    events = [
                        {'message': {'content': proof[:60]}, 'done': False},
                        {'message': {'content': proof[60:]}, 'done': True,
                         'done_reason': 'stop', 'eval_count': 30},
                    ]
                elif index == 3:
                    events = [{'message': {'content': json.dumps(critic)}, 'done': True,
                               'done_reason': 'stop', 'eval_count': 15}]
                elif index == 4:
                    events = [{'message': {'content': json.dumps(recorded)}, 'done': True,
                               'done_reason': 'stop', 'eval_count': 40}]
                elif index == 5:
                    events = [{'message': {'content': json.dumps(audit)}, 'done': True,
                               'done_reason': 'stop', 'eval_count': 20}]
                else:
                    events = [{'error': 'Unexpected extra inference call', 'done': True}]
                for event in events:
                    self.wfile.write(json.dumps(event).encode() + b'\n')
                    self.wfile.flush()

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host = f'http://127.0.0.1:{server.server_port}'
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'identity.tex').write_text(statement, encoding='utf-8')
                base = [sys.executable, '-m', 'mathagent', '--workspace', str(root), '--host', host]
                process = subprocess.run(base + ['--model', model, '--ctx', '16384', '--predict', '1024',
                    '--proof-rounds', '2', '--proof-tokens', '4096', '--proof-seconds', '30',
                    '--proof-file', 'identity.tex', '--prompt', 'Prove the statement in identity.tex.'],
                    capture_output=True, text=True, timeout=20)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assertIn('candidate_complete', process.stdout)
                self.assertIn('read_file', process.stdout)
                self.assertEqual(endpoints, [('GET', '/api/tags')] + [('POST', '/api/chat')] * 5)
                self.assertEqual(len(requests), 5)
                self.assertTrue(all(payload['model'] == model for payload in requests))
                self.assertTrue(all(payload['stream'] for payload in requests))
                self.assertTrue(requests[0]['think'])
                self.assertTrue(requests[1]['think'])
                self.assertNotIn('write_file', [tool['function']['name'] for tool in requests[0]['tools']])
                self.assertIn('read_proof_claim', [tool['function']['name'] for tool in requests[0]['tools']])
                self.assertIn(statement.rstrip(), requests[1]['messages'][-1]['content'])
                self.assertIn('PINNED ORIGINAL SOURCE', requests[1]['messages'][-1]['content'])
                for payload in requests[2:]:
                    self.assertFalse(payload['think'])
                    self.assertNotIn('tools', payload)
                    self.assertEqual(payload['format']['type'], 'object')
                self.assertEqual(set(requests[2]['format']['required']), set(critic))
                self.assertEqual(set(requests[3]['format']['required']), set(recorded))
                self.assertEqual(set(requests[3]['format']['properties']['claims']['items']['required']),
                                 set(review))
                self.assertEqual(set(requests[4]['format']['required']), set(audit))
                self.assertIn('CURRENT CANDIDATE', requests[2]['messages'][1]['content'])
                self.assertNotIn('FRESH INDEPENDENT CRITIQUE', requests[2]['messages'][1]['content'])
                self.assertIn('FRESH INDEPENDENT CRITIQUE', requests[3]['messages'][1]['content'])

                jobs = list((root / '.mathagent' / 'proofs').iterdir())
                self.assertEqual(len(jobs), 1)
                directory = jobs[0]
                state = json.loads((directory / 'state.json').read_text())
                self.assertEqual(state['status'], 'candidate_complete')
                self.assertEqual(state['rounds_started'], 1)
                self.assertEqual(state['tokens_charged'], 115)
                self.assertEqual([call['role'] for call in state['calls']],
                                 ['solver', 'solver', 'critic', 'recorder', 'auditor'])
                self.assertTrue(all(call['status'] == 'complete' for call in state['calls']))
                self.assertEqual(state['sources'][0]['content'], statement)
                self.assertEqual(state['claims'][0]['status'], 'reviewed')
                self.assertEqual(state['final_audit']['verdict'], 'complete')
                report = (directory / 'report.md').read_text()
                self.assertIn(proof, report)
                self.assertIn('No result in this report is a formal proof certificate.', report)
                self.assertTrue((directory / 'ledger.md').is_file())
                self.assertGreater(len(list((directory / 'artifacts').iterdir())), 4)

                # Stop the only HTTP server. Inspection and a completed-job
                # resume must remain usable without any running model server.
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
                for command in ('/proofs', f'/ledger {directory.name}'):
                    offline = subprocess.run(base + ['--prompt', command],
                        capture_output=True, text=True, timeout=10)
                    self.assertEqual(offline.returncode, 0, offline.stdout + offline.stderr)
                    self.assertIn(directory.name, offline.stdout)
                    self.assertIn('candidate_complete', offline.stdout)
                (root / 'identity.tex').write_text('A changed theorem must not replace the saved source.', encoding='utf-8')
                resumed = subprocess.run(base + ['--resume', directory.name,
                    '--proof-rounds', '99', '--proof-tokens', '999999', '--proof-seconds', '9999'],
                    capture_output=True, text=True, timeout=10)
                self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
                self.assertIn('candidate_complete', resumed.stdout)
                self.assertIn('changed or is unavailable', resumed.stdout)
                after = json.loads((directory / 'state.json').read_text())
                for field in ('settings', 'tokens_charged', 'rounds_started', 'sources', 'calls'):
                    self.assertEqual(after[field], state[field], field)
        finally:
            if thread.is_alive():
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
