"""Visual interface server: security boundaries and workflows against a simulated Ollama."""
import http.client
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from mathagent import cli
from mathagent.agent import Ollama
from mathagent.backends import LlamaCpp, OpenAICompatible
from mathagent.gui import server, store
from mathagent.gui.hub import Hub, Task
from mathagent.gui.watch import Watcher
from mathagent.ledger import ProofStore

MODEL = 'test-model:latest'


def text(content, eval_count=20, chunks=1):
    size = max(1, len(content) // chunks)
    parts = [content[i:i + size] for i in range(0, len(content), size)] or ['']
    events = [{'message': {'content': part}, 'done': False} for part in parts[:-1]]
    return events + [{'message': {'content': parts[-1]}, 'done': True, 'done_reason': 'stop', 'eval_count': eval_count}]


def tool(name, arguments):
    return [{'message': {'content': '', 'tool_calls': [{'function': {'name': name, 'arguments': arguments}}]},
             'done': True, 'done_reason': 'stop', 'eval_count': 8}]


class FakeOllama:
    """Scripted /api/chat replies; a (events, delay) tuple streams slowly."""

    def __init__(self):
        self.replies, self.requests = [], []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                body = json.dumps({'models': [{'name': MODEL}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                fake.requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                reply = fake.replies.pop(0) if fake.replies else text('Unexpected extra call.')
                events, delay = reply if isinstance(reply, tuple) else (reply, 0)
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson')
                self.end_headers()
                for event in events:
                    if delay:
                        time.sleep(delay)
                    try:
                        self.wfile.write(json.dumps(event).encode() + b'\n')
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host = f'http://127.0.0.1:{self.server.server_port}'

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class GuiCase(unittest.TestCase):
    extra_args = ()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.fake = FakeOllama()
        self.addCleanup(self.fake.close)
        args = cli.parser().parse_args(['--workspace', str(self.root), '--host', self.fake.host, '--model', MODEL,
                                        '--ctx', '16384', '--predict', '1024', *self.extra_args])
        args.workspace = self.root
        self.token = server.workspace_token(self.root)
        self.hub = Hub(args)
        self.server = server.bind('127.0.0.1', 0, self.hub, self.token)
        self.watcher = Watcher(self.root, self.hub.bus, interval=0.05)
        self.watcher.start()
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]
        self.addCleanup(self.stop)

    def stop(self):
        self.hub.shutdown(5)
        self.watcher.stop()
        self.server.shutdown()
        self.server.server_close()

    def request(self, method, path, body=None, headers=None, auth=True, raw=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        sent = {'Host': f'127.0.0.1:{self.port}'}
        if auth:
            sent['Authorization'] = 'Bearer ' + self.token
        data = raw
        if body is not None:
            data = json.dumps(body).encode()
            sent['Content-Type'] = 'application/json'
        sent.update(headers or {})
        connection.request(method, path, body=data, headers=sent)
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        try:
            value = json.loads(payload) if payload else {}
        except ValueError:
            value = payload.decode(errors='replace')
        return response.status, value, dict(response.getheaders())

    def ok(self, method, path, body=None):
        status, value, _ = self.request(method, path, body)
        self.assertEqual(status, 200, value)
        return value

    def wait_task(self, states=('done', 'paused', 'error'), timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            task = self.ok('GET', '/api/task')['task']
            if task and task['state'] in states:
                return task
            time.sleep(0.05)
        self.fail(f'task did not reach {states}: {task}')

    def wait_for(self, predicate, timeout=10, message='condition'):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.05)
        self.fail('timed out waiting for ' + message)


class SecurityTests(GuiCase):
    def test_api_requires_the_workspace_token(self):
        status, value, headers = self.request('GET', '/api/status', auth=False)
        self.assertEqual(status, 401)
        self.assertIn('access token', value['error'])
        self.assertIn("default-src 'self'", headers['Content-Security-Policy'])
        status, _, _ = self.request('GET', '/api/status', headers={'Authorization': 'Bearer wrong-token'}, auth=False)
        self.assertEqual(status, 401)
        status, value, headers = self.request('GET', '/api/status')
        self.assertEqual(status, 200)
        self.assertEqual(value['model'], MODEL)
        self.assertFalse(value['online'])
        self.assertTrue(value['ollama']['reachable'])
        self.assertNotIn('pair_urls', value)
        self.assertEqual(headers['Cache-Control'], 'no-store')

    def test_cookie_login_and_token_in_address_is_removed(self):
        status, _, _ = self.request('POST', '/api/login', {'token': 'nope'}, auth=False)
        self.assertEqual(status, 401)
        status, _, headers = self.request('POST', '/api/login', {'token': self.token}, auth=False)
        self.assertEqual(status, 200)
        cookie = headers['Set-Cookie']
        for flag in ('HttpOnly', 'SameSite=Strict', 'Path=/'):
            self.assertIn(flag, cookie)
        value = cookie.split(';')[0]
        status, _, _ = self.request('GET', '/api/proofs', headers={'Cookie': value}, auth=False)
        self.assertEqual(status, 200)
        status, _, headers = self.request('GET', '/?token=' + self.token, auth=False)
        self.assertEqual(status, 303)
        self.assertEqual(headers['Location'], '/')
        self.assertIn('HttpOnly', headers['Set-Cookie'])

    def test_rebinding_cross_origin_and_non_json_requests_are_refused(self):
        status, _, _ = self.request('GET', '/api/status', headers={'Host': 'attacker.example:8765'})
        self.assertEqual(status, 403)
        status, _, _ = self.request('POST', '/api/chats', {'mode': 'critic'}, headers={'Origin': 'http://attacker.example'})
        self.assertEqual(status, 403)
        status, _, _ = self.request('POST', '/api/chats', {'mode': 'critic'}, headers={'Sec-Fetch-Site': 'cross-site'})
        self.assertEqual(status, 403)
        status, _, _ = self.request('POST', '/api/chats', raw=b'mode=critic',
                                    headers={'Content-Type': 'application/x-www-form-urlencoded'})
        self.assertEqual(status, 415)
        status, _, _ = self.request('POST', '/api/chats', {'mode': 'critic'},
                                    headers={'Origin': f'http://127.0.0.1:{self.port}', 'Sec-Fetch-Site': 'same-origin'})
        self.assertEqual(status, 200)

    def test_rejected_bodies_close_the_connection_and_bad_json_is_refused(self):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        body = b'{"mode": "critic"}GET /api/status HTTP/1.1\r\nHost: x\r\n\r\n'
        connection.request('POST', '/api/chats', body=body, headers={'Host': f'127.0.0.1:{self.port}',
                           'Content-Type': 'application/json', 'Content-Length': str(len(body))})
        response = connection.getresponse()
        self.assertEqual(response.status, 401)
        self.assertEqual(response.getheader('Connection'), 'close')
        connection.close()
        deep = ('[' * 100000 + ']' * 100000).encode()
        status, value, _ = self.request('POST', '/api/chats', raw=deep, headers={'Content-Type': 'application/json'})
        self.assertEqual((status, value['error']), (400, 'Invalid JSON body'))
        status, _, headers = self.request('GET', '/%5Cattacker.example?token=' + self.token, auth=False)
        self.assertEqual((status, headers['Location']), (303, '/'))
        self.assertTrue(headers['Set-Cookie'].startswith(f'square_token_{self.port}='))

    def test_static_files_never_escape_the_asset_folder(self):
        for path in ('/../server.py', '/assets/../../hub.py', '/.hidden', '/%2e%2e/hub.py'):
            status, _, _ = self.request('GET', path, auth=False)
            self.assertEqual(status, 404, path)
        status, body, headers = self.request('GET', '/', auth=False)
        self.assertEqual(status, 200)
        self.assertIn('text/html', headers['Content-Type'])

    def test_artifact_and_job_paths_are_validated(self):
        self.assertEqual(self.request('GET', '/api/proofs/not-an-id')[0], 404)
        self.assertEqual(self.request('GET', '/api/proofs/00000000-0000-0000-0000-000000000000')[0], 404)
        self.assertEqual(self.request('GET', '/api/chats/00000000-0000-0000-0000-000000000000')[0], 404)
        self.assertEqual(self.request('POST', '/api/approvals/' + '0' * 32, {'approve': True})[0], 404)

    def test_token_file_is_private_and_reused(self):
        path = self.root / '.mathagent' / 'gui-token'
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(server.workspace_token(self.root), self.token)
        os.chmod(path, 0o644)
        replaced = server.workspace_token(self.root)
        self.assertNotEqual(replaced, self.token)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_browser_cannot_enable_literature_tools_beyond_launch(self):
        status, value, _ = self.request('POST', '/api/proofs', {'goal': 'Prove it.', 'literature': True})
        self.assertEqual(status, 400)
        self.assertIn('--proof-literature', value['error'])
        self.assertEqual(self.fake.requests, [])

    def test_gui_flag_rejects_one_shot_options(self):
        process = subprocess.run([sys.executable, '-m', 'mathagent', '--gui', '--prompt', '/proofs',
                                  '--workspace', str(self.root)], capture_output=True, text=True, timeout=20)
        self.assertEqual(process.returncode, 2)
        self.assertIn('--gui cannot be combined', process.stderr)


PROOF = ('Let x be an arbitrary real number. Equality is reflexive, so $x=x$. '
         'Since x was arbitrary, the statement holds for every real number.')
CRITIC = {'valid_steps': ['Reflexivity gives $x=x$ for arbitrary real x.'], 'first_invalid_step': '',
          'reason': '', 'missing_work': '', 'complete_candidate': True}
CLAIM = {'critical_claim': 'For every real number x, x=x.', 'assumptions': ['x is real.'], 'dependencies': [],
         'argument': PROOF, 'disposition': 'supported', 'objection': '', 'evidence': 'Reflexivity.',
         'next_task': 'Audit the complete proof.', 'new_progress': True, 'complete_candidate': False,
         'resolves': [], 'resolution': ''}
RECORD = {'claims': [CLAIM], 'complete_candidate': True, 'next_task': 'Audit the complete proof.',
          'strategy_summary': 'Apply reflexivity to an arbitrary real number.'}
AUDIT = {'verdict': 'complete', 'explanation': 'Reflexivity covers every real x.', 'objection': '', 'next_task': ''}


class ProofWorkflowTests(GuiCase):
    def test_start_runs_to_an_audited_candidate_and_is_inspectable(self):
        (self.root / 'identity.tex').write_text('Prove that every real $x$ satisfies $x=x$.\n')
        # A complete candidate the critic accepts goes straight to the whole-proof audit.
        self.fake.replies = [tool('read_file', {'path': 'identity.tex'}), text(PROOF, chunks=4),
                             text(json.dumps(CRITIC)), text(json.dumps(AUDIT))]
        started = self.ok('POST', '/api/proofs', {'goal': 'Prove the statement in identity.tex.',
                                                  'source_files': ['identity.tex'], 'rounds': 2,
                                                  'tokens': 8000, 'seconds': 60})
        proof_id = started['id']
        self.assertRegex(proof_id, r'^[0-9a-f-]{36}$')
        self.assertEqual(started['task']['label'], 'Proof')
        task = self.wait_task()
        self.assertEqual(task['state'], 'done', task)
        self.assertEqual(task['result_status'], 'candidate_complete')

        detail = self.ok('GET', f'/api/proofs/{proof_id}')
        self.assertEqual(detail['status'], 'candidate_complete')
        self.assertFalse(detail['running'])
        self.assertEqual(detail['candidate'], PROOF)
        self.assertEqual(detail['claims'][0]['status'], 'reviewed')
        # The interface labels these: no Record step, and the evidence is the auditor's answer.
        self.assertTrue(detail['rounds'][0]['fast_path'])
        self.assertEqual(detail['claims'][0]['record_kind'], 'whole_candidate_audit')
        self.assertEqual(detail['sources'], [{'path': 'identity.tex', 'sha256': detail['sources'][0]['sha256'], 'lines': 1}])
        self.assertEqual([c['role'] for c in detail['calls']], ['solver', 'solver', 'critic', 'auditor'])
        self.assertGreater(len(detail['artifacts']), 5)
        listed = self.ok('GET', '/api/proofs')['jobs']
        self.assertEqual([job['id'] for job in listed], [proof_id])
        self.assertEqual(listed[0]['claims'], {'reviewed': 1})
        report = self.ok('GET', f'/api/proofs/{proof_id}/report')
        self.assertIn('formal proof certificate', report['report'])
        self.assertTrue(report['ledger'])
        sources = self.ok('GET', f'/api/proofs/{proof_id}/sources')['sources']
        self.assertIn('x=x', sources[0]['content'])

        stream_name = detail['calls'][1]['stream']
        chunk = self.ok('GET', f'/api/proofs/{proof_id}/stream/{stream_name}?from=0')
        self.assertEqual(chunk['text'], PROOF)
        self.assertTrue(chunk['done'])
        later = self.ok('GET', f'/api/proofs/{proof_id}/stream/{stream_name}?from={chunk["end"]}')
        self.assertEqual((later['text'], later['start']), ('', chunk['end']))
        request = self.ok('GET', f'/api/proofs/{proof_id}/artifacts/{detail["calls"][0]["request"]}')
        self.assertEqual(request['json']['model'], MODEL)
        status, _, _ = self.request('GET', f'/api/proofs/{proof_id}/artifacts/..%2Fstate.json')
        self.assertEqual(status, 404)
        status, _, _ = self.request('GET', f'/api/proofs/{proof_id}/artifacts/state.json')
        self.assertEqual(status, 400)

    def test_pause_saves_a_checkpoint_and_resume_keeps_the_budget(self):
        slow = [{'message': {'thinking': 'step '}, 'done': False}] * 200 + text('Unfinished.')
        self.fake.replies = [(slow, 0.02)]
        started = self.ok('POST', '/api/proofs', {'goal': 'Prove that 1 = 1.', 'rounds': 2, 'tokens': 6000, 'seconds': 120})
        proof_id = started['id']
        self.wait_for(lambda: self.ok('GET', f'/api/proofs/{proof_id}')['live'], message='a live solver call')
        status, value, _ = self.request('POST', '/api/proofs', {'goal': 'Another proof.'})
        self.assertEqual(status, 409)
        self.assertIn('one task at a time', value['error'])
        self.assertEqual(self.ok('POST', '/api/task/pause')['task']['state'], 'pausing')
        task = self.wait_task()
        self.assertEqual(task['state'], 'paused')
        detail = self.ok('GET', f'/api/proofs/{proof_id}')
        self.assertEqual(detail['status'], 'paused')
        self.assertEqual(detail['calls'][-1]['status'], 'interrupted')
        charged = detail['budget']['tokens']['used']
        self.assertGreater(charged, 0)
        self.assertLess(len(self.fake.requests), 3)

        self.fake.replies = [text(json.dumps(CRITIC)), text(json.dumps({**RECORD, 'claims': [], 'complete_candidate': False})),
                             text('Still partial.'), text(json.dumps(CRITIC)), text(json.dumps({**RECORD, 'claims': []}))]
        resumed = self.ok('POST', f'/api/proofs/{proof_id}/resume')
        self.assertEqual(resumed['id'], proof_id)
        task = self.wait_task(timeout=30)
        self.assertIn(task['state'], ('done', 'paused'))
        after = self.ok('GET', f'/api/proofs/{proof_id}')
        self.assertGreaterEqual(after['budget']['tokens']['used'], charged)
        self.assertEqual(after['settings']['max_tokens'], 6000)

    def test_invalid_budgets_are_rejected_before_any_model_call(self):
        for body in ({'goal': ''}, {'goal': 'x', 'rounds': 0}, {'goal': 'x', 'rounds': True},
                     {'goal': 'x', 'seconds': -1}, {'goal': 'x', 'source_files': ['../escape.tex']},
                     {'goal': 'x', 'ctx': 1024}):
            status, value, _ = self.request('POST', '/api/proofs', body)
            self.assertEqual(status, 400, (body, value))
        self.assertEqual(self.fake.requests, [])


class ChatTests(GuiCase):
    def test_turn_with_an_approved_write_is_saved(self):
        self.fake.replies = [tool('write_file', {'path': 'notes.md', 'content': 'Checked.\n'}),
                             text('I wrote the note. Here $a^2 \\ge 0$.', chunks=3)]
        chat = self.ok('POST', '/api/chats', {'mode': 'critic'})
        self.assertEqual((chat['mode'], chat['transcript']), ('critic', []))
        task = self.ok('POST', f'/api/chats/{chat["id"]}/messages', {'content': 'Write a note.'})['task']
        self.assertEqual(task['label'], 'Critique')
        approval = self.wait_for(lambda: self.ok('GET', '/api/task')['approvals'], message='an approval')[0]
        self.assertEqual(approval['kind'], 'write')
        self.assertIn('+Checked.', approval['preview'])
        self.assertFalse((self.root / 'notes.md').exists())
        self.ok('POST', f'/api/approvals/{approval["id"]}', {'approve': True})
        self.assertEqual(self.wait_task()['state'], 'done')
        self.assertEqual((self.root / 'notes.md').read_text(), 'Checked.\n')
        saved = self.ok('GET', f'/api/chats/{chat["id"]}')
        roles = [item['role'] for item in saved['transcript']]
        self.assertEqual(roles, ['user', 'assistant', 'tool', 'assistant'])
        self.assertEqual(saved['transcript'][1]['tool_calls'][0]['function']['name'], 'write_file')
        self.assertIn('Wrote notes.md', saved['transcript'][2]['content'])
        self.assertEqual(saved['transcript'][3]['content'], 'I wrote the note. Here $a^2 \\ge 0$.')
        self.assertEqual(saved['title'], 'Write a note.')
        self.assertEqual(self.ok('GET', '/api/chats')['chats'][0]['messages'], 1)

        self.fake.replies = [text('The previous answer is fine.')]
        self.ok('POST', f'/api/chats/{chat["id"]}/review')
        self.assertEqual(self.wait_task()['state'], 'done')
        saved = self.ok('GET', f'/api/chats/{chat["id"]}')
        self.assertEqual(saved['transcript'][-1]['role'], 'review')
        self.assertEqual(saved['context_messages'], 6)
        self.assertNotIn('tools', self.fake.requests[-1]['messages'][0])
        self.assertIn('Audit this proposed answer', self.fake.requests[-1]['messages'][-1]['content'])

    def test_denied_write_changes_nothing(self):
        self.fake.replies = [tool('write_file', {'path': 'notes.md', 'content': 'x'}), text('Understood.')]
        chat = self.ok('POST', '/api/chats', {'mode': 'explore'})
        self.ok('POST', f'/api/chats/{chat["id"]}/messages', {'content': 'Write it.'})
        approval = self.wait_for(lambda: self.ok('GET', '/api/task')['approvals'], message='an approval')[0]
        self.ok('POST', f'/api/approvals/{approval["id"]}', {'approve': False})
        self.assertEqual(self.wait_task()['state'], 'done')
        self.assertFalse((self.root / 'notes.md').exists())
        saved = self.ok('GET', f'/api/chats/{chat["id"]}')
        self.assertIn('User denied the write', saved['transcript'][2]['content'])

    def test_paused_turn_stays_visible_but_out_of_the_model_context(self):
        slow = [{'message': {'content': 'word '}, 'done': False}] * 200 + text('end')
        self.fake.replies = [(slow, 0.02)]
        chat = self.ok('POST', '/api/chats', {'mode': 'critic'})
        self.ok('POST', f'/api/chats/{chat["id"]}/messages', {'content': 'A long question.'})
        live = self.wait_for(lambda: (self.ok('GET', '/api/task')['live'] or {}).get('steps'), message='live output')
        self.assertEqual(live[0]['type'], 'call')
        self.ok('POST', '/api/task/pause')
        self.assertEqual(self.wait_task()['state'], 'paused')
        saved = self.ok('GET', f'/api/chats/{chat["id"]}')
        self.assertEqual(saved['context_messages'], 0)
        self.assertEqual(saved['transcript'][0], {**saved['transcript'][0], 'role': 'user', 'discarded': True})
        self.assertIn('paused', saved['transcript'][1]['content'])

    def test_chat_mode_and_message_are_validated(self):
        self.assertEqual(self.request('POST', '/api/chats', {'mode': 'prove'})[0], 400)
        chat = self.ok('POST', '/api/chats', {'mode': 'critic', 'think': False})
        self.assertFalse(chat['settings']['think'])
        self.assertEqual(self.request('POST', f'/api/chats/{chat["id"]}/messages', {'content': '   '})[0], 400)
        self.assertEqual(self.request('POST', f'/api/chats/{chat["id"]}/review')[0], 400)
        updated = self.ok('POST', f'/api/chats/{chat["id"]}/settings', {'think': True})
        self.assertTrue(updated['settings']['think'])


class ResearchTests(GuiCase):
    def test_offline_referee_report_with_manuscript_evidence(self):
        (self.root / 'manuscript.tex').write_text('Claim: for every real x, x=x.\nProof: equality is reflexive.\n')
        draft = '# Referee report\n\nThe manuscript states reflexivity [M1:L1-L2].\n'
        self.fake.replies = [
            text('Plan: read the manuscript.'),
            [{'message': {'content': '', 'tool_calls': [{'function': {'name': 'read_manuscript',
              'arguments': {'source_id': 'M1', 'start_line': 1, 'end_line': 2}}}]},
              'done': True, 'done_reason': 'stop', 'eval_count': 20, 'prompt_eval_count': 400}],
            text('Read; reflexivity is the argument.'), text(draft),
            text('The cited passage supports the description.'), text(draft)]
        started = self.ok('POST', '/api/research', {'kind': 'referee', 'goal': 'Review this manuscript.',
                                                    'source_files': ['manuscript.tex'], 'rounds': 2, 'tokens': 16000,
                                                    'input_tokens': 50000, 'requests': 0, 'seconds': 60})
        job_id = started['id']
        self.assertEqual(started['task']['label'], 'Referee report')
        self.assertEqual(self.wait_task(timeout=30)['state'], 'done')
        detail = self.ok('GET', f'/api/research/{job_id}')
        self.assertEqual(detail['status'], 'reviewed', detail['citation_issues'])
        self.assertEqual(detail['manuscript_ranges'], {'M1': [[1, 2]]})
        self.assertEqual(detail['evidence'][0]['tool'], 'read_manuscript')
        self.assertIn('[M1:L1-L2]', detail['draft'])
        self.assertEqual(detail['budget']['requests'], {'used': 0, 'limit': 0})
        self.assertEqual(self.ok('GET', '/api/research')['jobs'][0]['kind'], 'referee')
        sources = self.ok('GET', f'/api/research/{job_id}/sources')['sources']
        self.assertEqual(sources[0]['id'], 'M1')
        stream = detail['calls'][0]['stream']
        self.assertEqual(self.ok('GET', f'/api/research/{job_id}/stream/{stream}?from=0')['text'], 'Plan: read the manuscript.')
        self.assertTrue(all(request['think'] is False for request in self.fake.requests))


class LiveUpdateTests(GuiCase):
    def read_events(self, count, headers=None, timeout=5, sink=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=timeout)
        connection.request('GET', '/api/events', headers={'Host': f'127.0.0.1:{self.port}',
                           'Authorization': 'Bearer ' + self.token, **(headers or {})})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn('text/event-stream', response.getheader('Content-Type'))
        events, last_id = ([] if sink is None else sink), None
        while len(events) < count:
            line = response.fp.readline().decode().strip()
            if line.startswith('id: '):
                last_id = line[4:]
            elif line.startswith('data: '):
                events.append(json.loads(line[6:]))
        connection.close()
        return events, last_id

    def test_event_stream_says_hello_and_replays_after_reconnect(self):
        received = []
        reader = threading.Thread(target=lambda: received.extend(self.read_events(2)[0]), daemon=True)
        reader.start()
        time.sleep(0.3)
        chat = self.ok('POST', '/api/chats', {'mode': 'critic'})
        reader.join(5)
        self.assertEqual(received[0]['type'], 'hello')
        self.assertEqual(received[1], {**received[1], 'type': 'chat_saved', 'chat': chat['id']})
        instance = received[0]['instance']
        # A reconnect with an old event ID receives what it missed.
        second = self.ok('POST', '/api/chats', {'mode': 'explore'})
        events, _ = self.read_events(2, {'Last-Event-ID': f'{instance}:{received[1]["seq"]}'})
        self.assertEqual(events[1]['chat'], second['id'])
        events, _ = self.read_events(1, {'Last-Event-ID': 'another-server:5'})
        self.assertTrue(events[0]['resync'])

    def test_watcher_follows_a_job_started_elsewhere(self):
        # A terminal-started proof: running state, running call, streamed output.
        job = ProofStore.create(self.root, 'Prove it.', {'max_rounds': 1, 'max_tokens': 1000, 'max_seconds': 60})
        stream = job.start_stream('solver')
        job.state['status'] = 'running'
        job.state['calls'].append({'role': 'solver', 'round': 1, 'status': 'running', 'reserved_tokens': 100,
                                   'stream': stream, 'request': stream})
        job.save()
        seen = []

        def listen():
            try:
                self.read_events(50, timeout=10, sink=seen)
            except OSError:
                pass  # the test ends the stream by stopping the server
        reader = threading.Thread(target=listen, daemon=True)
        reader.start()
        time.sleep(0.3)
        job.append_stream(stream, {'message': {'thinking': 'Consider n.'}, 'done': False})
        job.append_stream(stream, {'message': {'content': 'Let $n$ be even.'}, 'done': False})
        self.wait_for(lambda: any(e['type'] == 'stream' and 'Let $n$' in e['text'] for e in seen), message='a stream event')
        streamed = [e for e in seen if e['type'] == 'stream']
        self.assertEqual(streamed[-1]['id'], job.state['id'])
        self.assertEqual(streamed[-1]['role'], 'solver')
        self.assertEqual(''.join(e['thinking'] for e in streamed), 'Consider n.')
        detail = self.ok('GET', f'/api/proofs/{job.state["id"]}')
        self.assertFalse(detail['running'])  # no process holds the run lock
        with job.lock():
            self.assertTrue(self.ok('GET', f'/api/proofs/{job.state["id"]}')['running'])


class BackendTests(unittest.TestCase):
    """The interface builds the same model client as the terminal, for every --backend."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def hub(self, *extra):
        return Hub(cli.parser().parse_args(['--workspace', str(self.root), '--host', 'http://127.0.0.1:9',
                                            '--model', MODEL, *extra]))

    def test_each_backend_gets_a_pausable_client_with_the_sampling_settings(self):
        for backend, base in (('ollama', Ollama), ('openai', OpenAICompatible), ('llamacpp', LlamaCpp)):
            with self.subTest(backend=backend):
                hub = self.hub('--backend', backend, '--request-timeout', '42', '--seed', '7',
                               '--temperature', '0.3', '--top-p', '0.9')
                task = Task('proof', '', 'Proof')
                agent, _ = hub._engine(task)
                self.assertIsInstance(agent.client, base)
                self.assertEqual((agent.client.backend, agent.client.timeout), (backend, 42))
                self.assertEqual((agent.seed, agent.temperature, agent.top_p), (7, 0.3, 0.9))
                task.cancel.set()
                with self.assertRaises(KeyboardInterrupt):
                    next(agent.client.stream({'model': MODEL, 'messages': [],
                                              'options': {'num_ctx': 4096, 'num_predict': 64}}))

    def test_model_status_speaks_the_backend_protocol(self):
        class Models(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                found = self.path == '/v1/models'
                body = json.dumps({'data': [{'id': MODEL}]} if found else {}).encode()
                self.send_response(200 if found else 404)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        httpd = ThreadingHTTPServer(('127.0.0.1', 0), Models)
        threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        hub = self.hub('--backend', 'openai', '--host', f'http://127.0.0.1:{httpd.server_address[1]}')
        self.assertEqual(hub.models(), {'reachable': True, 'models': [MODEL]})

    def test_gui_flag_rejects_parallel_and_cooperative_proofs(self):
        for extra in (['--proof-workers', '2'], ['--proof-strategy', 'cooperative']):
            process = subprocess.run([sys.executable, '-m', 'mathagent', '--gui', *extra, '--workspace', str(self.root)],
                                     capture_output=True, text=True, timeout=20)
            self.assertEqual(process.returncode, 2, extra)
            self.assertIn('one proof job at a time', process.stderr)


class StoreTests(unittest.TestCase):
    def test_parse_stream_ignores_a_partial_final_line(self):
        value = store.parse_stream('{"message": {"content": "a"}}\n{"message": {"thinking": "b"}, "done": true, "done_reason": "stop"}\n{"message"')
        self.assertEqual((value['text'], value['thinking'], value['done']), ('a', 'b', True))

    def test_chat_files_reject_foreign_modes_and_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            chat = store.create_chat(temporary, 'critic')
            path = Path(temporary) / '.mathagent' / 'chats' / (chat['id'] + '.json')
            self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
            data = json.loads(path.read_text())
            data['mode'] = 'prove'
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                store.load_chat(temporary, chat['id'])
            with self.assertRaises(ValueError):
                store.load_chat(temporary, '../../etc/passwd')
            self.assertEqual(store.list_chats(temporary), [])


if __name__ == '__main__':
    unittest.main()
