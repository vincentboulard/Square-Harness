import json
from pathlib import Path
import tempfile
import threading
import unittest
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mathagent.agent import Agent, AgentError, Ollama
from mathagent.tools import Workspace


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'lemma.tex').write_text('Claim\nA flawed proof\n')
        self.ws = Workspace(self.root)

    def test_read_search_and_invalid_args(self):
        self.assertIn('2: A flawed proof', self.ws.read_file('lemma.tex', 2, 2))
        self.assertIn('lemma.tex:2', self.ws.search_text('flawed'))
        self.assertIn('Tool error', self.ws.execute('read_file', {'path': 'lemma.tex', 'start_line': 'x'}))
        self.assertIn('Tool error', self.ws.execute('__class__', {}))

    def test_boundaries(self):
        for path in ('../outside.tex', '/etc/passwd', '.private.txt'):
            with self.assertRaises(ValueError):
                self.ws.path(path)
        (self.root / 'escape.txt').symlink_to('/etc/passwd')
        with self.assertRaises(ValueError):
            self.ws.path('escape.txt')
        (self.root / '.hidden.txt').write_text('secret')
        (self.root / 'alias.txt').symlink_to(self.root / '.hidden.txt')
        with self.assertRaises(ValueError):
            self.ws.path('alias.txt')
        self.assertNotIn('alias.txt', self.ws.list_files())

    def test_write_denied_then_approved(self):
        self.assertIn('denied', self.ws.write_file('new.tex', 'Hello'))
        self.assertFalse((self.root / 'new.tex').exists())
        previews = []
        self.ws.approve = lambda text: previews.append(text) or True
        self.ws.write_file('new.tex', 'Hello\n')
        self.assertEqual((self.root / 'new.tex').read_text(), 'Hello\n')
        self.assertIn('+Hello', previews[0])

    def test_exact_pins_do_not_enable_discovery_of_other_files(self):
        (self.root / 'other.tex').write_text('Unattached source\n')
        workspace = Workspace(self.root, read_types=[])
        workspace.pin_files(['lemma.tex'])
        self.assertIn('A flawed proof', workspace.execute('read_file', {'path': 'lemma.tex'}))
        self.assertIn('not allowed', workspace.execute('read_file', {'path': 'other.tex'}))
        self.assertEqual(workspace.list_files(), '(no readable files)')
        self.assertEqual(workspace.search_text('flawed'), '(no matches in supported files scanned)')
        for path in ('../outside.tex', '.private.txt', '/etc/passwd'):
            with self.assertRaises(ValueError):
                workspace.pin_files([path])

    def test_python_opt_in_and_approval(self):
        self.assertIn('disabled', self.ws.execute('run_python', {'code': 'print(1)'}))
        self.ws.allow_python = True
        self.assertIn('denied', self.ws.run_python('print(1)'))
        self.ws.approve = lambda text: True
        self.assertIn('42', self.ws.run_python('print(6*7)'))


class FakeClient:
    def __init__(self, events):
        self.events = iter(events)
        self.requests = []

    def stream(self, payload):
        self.requests.append(payload)
        yield from next(self.events)


def final(text='Answer', reason='stop'):
    return [{'message': {'content': text}, 'done': True, 'done_reason': reason}]


def call(name='list_files'):
    return [{'message': {'tool_calls': [{'function': {'name': name, 'arguments': {}}}]}, 'done': True}]


class LoopTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ws = Workspace(tmp.name)

    def test_budget_and_multi_tool_loop(self):
        client = FakeClient([call(), final()])
        agent = Agent(client, self.ws, ctx=16384, max_rounds=1)
        self.assertEqual(agent.run('Inspect files'), 'Answer')
        self.assertNotIn('tools', client.requests[1])
        self.assertEqual(agent.history[-2]['role'], 'tool')

    def test_incomplete_stream_does_not_commit(self):
        agent = Agent(FakeClient([[{'message': {'content': 'Partial'}}]]), self.ws)
        with self.assertRaises(AgentError):
            agent.run('Hi')
        self.assertEqual(agent.history, [])

    def test_length_stop_is_visible(self):
        events = []
        agent = Agent(FakeClient([final('Incomplete', 'length')]), self.ws)
        agent.run('Prove it', lambda kind, value: events.append((kind, value)))
        self.assertTrue(any('output limit' in value for kind, value in events))

    def test_old_turn_pruning_is_explicit(self):
        agent = Agent(FakeClient([final()]), self.ws)
        agent.history = [{'role': 'user', 'content': 'old' * 10000},
                         {'role': 'assistant', 'content': 'old response'}]
        events = []
        agent.run('New question', lambda kind, value: events.append(value))
        self.assertTrue(any('Dropped' in v for v in events))
        self.assertEqual(agent.history[0]['content'], 'New question')

    def test_current_turn_overflow_preserves_history(self):
        agent = Agent(FakeClient([]), self.ws)
        with self.assertRaises(AgentError):
            agent.run('x' * 100000)
        self.assertEqual(agent.history, [])

    def test_interrupt_preserves_history(self):
        class Interrupted:
            def stream(self, payload):
                raise KeyboardInterrupt
        agent = Agent(Interrupted(), self.ws)
        with self.assertRaises(KeyboardInterrupt):
            agent.run('Question')
        self.assertEqual(agent.history, [])


class HttpIntegration(unittest.TestCase):
    def test_real_http_ndjson_roundtrip(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({'models': [{'name': 'qwen3.8:27b'}]}).encode())

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(payload)
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson')
                self.end_headers()
                if len(requests) == 1:
                    events = [{'message': {'thinking': 'Inspect the file.'}, 'done': False},
                              {'message': {'tool_calls': [{'function': {'name': 'read_file',
                                 'arguments': {'path': 'lemma.tex'}}}]}, 'done': True}]
                else:
                    events = [{'message': {'content': 'The claim '}, 'done': False},
                              {'message': {'content': 'is false.'}, 'done': True, 'eval_count': 4}]
                for event in events:
                    self.wfile.write(json.dumps(event).encode() + b'\n')
                    self.wfile.flush()
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = Ollama(f'http://127.0.0.1:{server.server_port}')
            self.assertEqual(client.models(), ['qwen3.8:27b'])
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / 'lemma.tex').write_text('A false claim')
                agent = Agent(client, Workspace(tmp), ctx=16384)
                self.assertEqual(agent.run('Read lemma.tex'), 'The claim is false.')
                self.assertIn('A false claim', requests[1]['messages'][-1]['content'])
                self.assertEqual(requests[1]['messages'][-2]['thinking'], 'Inspect the file.')
                self.assertNotIn('thinking', agent.history[1])
                requests.clear()
                cli = subprocess.run([sys.executable, '-m', 'mathagent', '--host', client.host,
                    '--workspace', tmp, '--ctx', '16384', '--mode', 'critic', '--prompt', 'Read lemma.tex'],
                    capture_output=True, text=True, timeout=15)
                self.assertEqual(cli.returncode, 0, cli.stderr)
                self.assertIn('The claim is false.', cli.stdout)
                self.assertIn('read_file', cli.stdout)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
