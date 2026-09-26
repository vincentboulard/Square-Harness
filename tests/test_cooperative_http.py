"""The cooperative controller really overlaps distinct ready tasks over HTTP."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

from mathagent.agent import Agent, Ollama
from mathagent.cooperative import run_cooperative_proof
from mathagent.tools import Workspace
from test_cooperative import GOAL, PROOF, audit, contract, critic, event, plan


def original_request(payload):
    for message in payload['messages']:
        content = message.get('content', '')
        if 'ORIGINAL REQUEST (unchanged):\n' in content:
            return content.split('ORIGINAL REQUEST (unchanged):\n', 1)[1].split(
                '\nORIGINAL SOURCE SNAPSHOTS', 1)[0]
    return ''


def worker_task_id(payload):
    goal = original_request(payload)
    if 'EXACT TASK CONTRACT:\n' not in goal:
        return None
    return json.loads(goal.split('EXACT TASK CONTRACT:\n', 1)[1])['id']


class CooperativeHttpIntegration(unittest.TestCase):
    def test_independent_workers_overlap_then_dependency_and_original_goal_are_audited(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / 'workspace'
            workspace.mkdir()
            (workspace / 'statement.txt').write_text(GOAL + '\nORIGINAL_SOURCE_BOUNDARY: all real x,y.\n')
            (workspace / 'private-solution.txt').write_text('PRIVATE_SOLUTION_MUST_NOT_BE_COPIED')
            tasks = plan()['tasks'] + [contract('T3', statement=GOAL, dependencies=['T1', 'T2'])]
            advisor_plan = plan(tasks=tasks)
            proofs = {'T1': 'EVIDENCE_T1: In the real field, multiplication is commutative, so x*y=y*x.',
                      'T2': 'EVIDENCE_T2: Applying distributivity twice gives (x+y)^2=x*x+x*y+y*x+y*y.',
                      'T3': 'EVIDENCE_T3: ' + PROOF}
            barrier = threading.Barrier(2, timeout=10)
            overlap = threading.Event()
            lock = threading.Lock()
            requests, starts, finished_audits, handler_errors = [], [], set(), []
            active = maximum = 0

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_POST(self):
                    nonlocal active, maximum
                    try:
                        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                        with lock:
                            requests.append(payload)
                            active += 1
                            maximum = max(maximum, active)
                        fields = payload.get('format', {}).get('properties', {})
                        task_id = worker_task_id(payload)
                        if 'route' in fields:
                            value = advisor_plan
                        elif not fields:
                            if task_id:
                                with lock:
                                    starts.append((task_id, set(finished_audits), payload))
                                if task_id in {'T1', 'T2'}:
                                    barrier.wait()
                                    overlap.set()
                                value = proofs[task_id]
                            else:
                                value = PROOF
                        elif 'valid_steps' in fields:
                            value = critic()
                        elif 'verdict' in fields:
                            value = audit()
                            if task_id:
                                with lock:
                                    finished_audits.add(task_id)
                        else:
                            raise AssertionError('Unexpected cooperative request schema: ' + str(sorted(fields)))
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/x-ndjson')
                        self.end_headers()
                        self.wfile.write((json.dumps(event(value)) + '\n').encode())
                        self.wfile.flush()
                    except BaseException as exc:
                        with lock:
                            handler_errors.append(f'{type(exc).__name__}: {exc}')
                        self.send_error(500, 'Test model failed')
                    finally:
                        with lock:
                            active -= 1

            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                agent = Agent(Ollama(f'http://127.0.0.1:{server.server_port}'), Workspace(workspace),
                              model='test', ctx=32768, predict=256)
                result = run_cooperative_proof(agent, GOAL, output_dir=root / 'cooperative',
                    source_files=['statement.txt'], max_tokens=60000, max_seconds=45,
                    max_rounds=1, max_predict=512, concurrency=3, seed=7)
                self.assertEqual(handler_errors, [])
                self.assertTrue(overlap.is_set())
                self.assertGreaterEqual(maximum, 2)
                self.assertEqual({task_id for task_id, _, _ in starts}, {'T1', 'T2', 'T3'})
                dependent = next(item for item in starts if item[0] == 'T3')
                self.assertTrue({'T1', 'T2'} <= dependent[1])
                dependent_text = json.dumps(dependent[2])
                self.assertIn('EVIDENCE_T1', dependent_text)
                self.assertIn('EVIDENCE_T2', dependent_text)
                self.assertIn('No constants occur.', dependent_text)
                self.assertNotIn('PRIVATE_SOLUTION_MUST_NOT_BE_COPIED', json.dumps(requests))
                final_audit = requests[-1]
                self.assertEqual(original_request(final_audit), GOAL)
                self.assertIn(PROOF, json.dumps(final_audit))
                self.assertIn('ORIGINAL_SOURCE_BOUNDARY', json.dumps(final_audit))
                self.assertIn(tasks[0]['statement'], json.dumps(final_audit))
                self.assertIn('not supplied as proof here', json.dumps(final_audit).lower())
                self.assertNotIn('EVIDENCE_T1', json.dumps(final_audit))
                self.assertEqual(len(requests), 13)
                self.assertEqual(result['tokens_charged'], 130)
                self.assertEqual(result['request_count'], 13)
                self.assertEqual(Path(result['proof_path']).read_text(), PROOF)
                self.assertEqual(Path(result['answer_path']).read_text(), PROOF)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
