"""Real CLI/HTTP smoke test of a bounded offline referee-report workflow."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class ResearchHttpTests(unittest.TestCase):
    def test_offline_referee_reads_manuscript_and_exports_reviewed_markdown(self):
        requests = []
        draft = '# Referee report\n\nThe manuscript states reflexivity [M1:L1-L2].\n\n## Literature\nNo external search was performed; novelty is unassessed.\n'
        replies = [
            {'content': 'Read the manuscript, assess its argument, and report offline limitations.'},
            {'content': '', 'tool_calls': [{'function': {'name': 'read_manuscript',
                'arguments': {'source_id': 'M1', 'start_line': 1, 'end_line': 2}}}]},
            {'content': 'The source is read. Reflexivity is the stated argument; external comparisons remain unavailable.'},
            {'content': draft},
            {'content': 'The cited manuscript passage supports the description. External novelty remains unchecked; retain that limitation.'},
            {'content': draft},
        ]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps({'models': [{'name': 'qwen3.8:27b'}]}).encode())
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(payload)
                self.send_response(200); self.end_headers()
                index = len(requests)-1
                message = replies[index] if index < len(replies) else {'content': 'Unexpected extra model call.'}
                event = {'message': message, 'done': True, 'done_reason': 'stop',
                         'eval_count': 25, 'prompt_eval_count': 500}
                self.wfile.write(json.dumps(event).encode() + b'\n')
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'manuscript.tex').write_text('Claim: for every real x, x=x.\nProof: equality is reflexive.\n')
                command = [sys.executable, '-m', 'mathagent', '--workspace', tmp,
                    '--host', f'http://127.0.0.1:{server.server_port}', '--offline',
                    '--mode', 'referee', '--ctx', '16384', '--research-file', 'manuscript.tex',
                    '--research-rounds', '2', '--research-tokens', '16000',
                    '--research-input-tokens', '50000', '--research-requests', '0',
                    '--research-seconds', '30', '--output', 'referee.md', '--prompt', 'Review this manuscript.']
                proc = subprocess.run(command, capture_output=True, text=True, timeout=30,
                                      cwd=Path(__file__).resolve().parents[1])
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertEqual(len(requests), 6, proc.stdout)
                report = (root / 'referee.md').read_text()
                self.assertIn('[M1:L1-L2]', report)
                self.assertIn('novelty is unassessed', report)
                state_path = next((root / '.mathagent/research').glob('*/state.json'))
                state = json.loads(state_path.read_text())
                self.assertIn('(' + state_path.parent.relative_to(root).as_posix() + '/artifacts/', report)
                self.assertEqual(state['status'], 'reviewed', proc.stdout)
                self.assertEqual(state['literature_snapshot']['stats']['requests'], 0)
                self.assertTrue(all(request['think'] is False for request in requests))
                inspected = subprocess.run([sys.executable, '-m', 'mathagent', '--workspace', tmp,
                    '--prompt', '/research-report ' + state['id']], capture_output=True, text=True,
                    timeout=10, cwd=Path(__file__).resolve().parents[1])
                self.assertEqual(inspected.returncode, 0, inspected.stdout + inspected.stderr)
                self.assertIn('novelty is unassessed', inspected.stdout)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
