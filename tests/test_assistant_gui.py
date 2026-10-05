"""The Assistant answers and delegates inside one saved, bounded GUI turn."""
import json
import hashlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

try:
    from .test_gui import GuiCase, text, tool, PROOF, APPROVAL
except ImportError:
    from test_gui import GuiCase, text, tool, PROOF, APPROVAL

from mathagent.gui import store
from mathagent.worker_errors import WorkerInputError


def delegation(mode, request, files=(), effort='low'):
    return tool('delegate', {'mode': mode, 'request': request, 'files': list(files),
                             'effort': effort, 'reason': 'An independent check will help.'})


class FakeOpenAI:
    """The GUI fixture's scripted replies exposed through actual SSE/vLLM APIs."""

    def __init__(self):
        self.replies, self.requests = [], []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                body = json.dumps({'data': [{'id': 'test-model:latest'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/tokenize':
                    body = json.dumps({'count': 200, 'max_model_len': 40960}).encode()
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                fake.requests.append(payload)
                reply = fake.replies.pop(0) if fake.replies else text('Unexpected extra call.')
                events, delay = reply if isinstance(reply, tuple) else (reply, 0)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                for event in events:
                    if delay:
                        time.sleep(delay)
                    message = event.get('message', {})
                    delta = {key: message[key] for key in ('content',) if key in message}
                    if message.get('thinking'):
                        delta['reasoning_content'] = message['thinking']
                    if message.get('tool_calls'):
                        delta['tool_calls'] = [{'index': index, 'id': f'tool_{index}', 'type': 'function',
                                               'function': {'name': call['function']['name'],
                                                            'arguments': json.dumps(call['function']['arguments'])}}
                                              for index, call in enumerate(message['tool_calls'])]
                    output = [{'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]}]
                    if event.get('done'):
                        output += [{'choices': [{'index': 0, 'delta': {}, 'finish_reason':
                                                'tool_calls' if message.get('tool_calls') else event.get('done_reason', 'stop')}]},
                                   {'choices': [], 'usage': {'prompt_tokens': event.get('prompt_eval_count', 200),
                                                            'completion_tokens': event.get('eval_count', 20)}}]
                    try:
                        for item in output:
                            self.wfile.write(('data: ' + json.dumps(item) + '\n\n').encode())
                        if event.get('done'):
                            self.wfile.write(b'data: [DONE]\n\n')
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


class AssistantGuiTests(GuiCase):
    extra_args = ('--ctx', '40960')

    def conversation(self, **settings):
        return self.ok('POST', '/api/chats', {'mode': 'free', **settings})['id']

    def ask(self, chat, content, files=()):
        return self.ok('POST', f'/api/chats/{chat}/route',
                       {'content': content, 'files': list(files)})

    def detail(self, chat):
        return self.ok('GET', f'/api/chats/{chat}')

    def openai_fixture(self):
        self.fake.close()
        self.fake = FakeOpenAI()
        self.addCleanup(self.fake.close)
        self.hub.args.backend, self.hub.args.host = 'openai', self.fake.host

    def test_elementary_answer_needs_one_nonthinking_call_and_no_worker(self):
        chat = self.conversation()
        answer = 'If a is nonnegative, a*a is nonnegative. Otherwise (-a)*(-a)=a*a is positive.'
        self.fake.replies = [text(answer, eval_count=37)]
        self.ask(chat, 'Prove that a^2 >= 0 for every real a.')
        task = self.wait_task()
        self.assertEqual(task['state'], 'done', task)
        self.assertEqual((task['kind'], task['target']), ('chat', chat))
        detail = self.detail(chat)
        self.assertEqual(detail['assistant_status'], 'complete')
        self.assertEqual([item['content'] for item in detail['transcript'] if item['role'] == 'assistant'][-1], answer)
        self.assertFalse(any(item['role'] == 'route' for item in detail['transcript']))
        self.assertEqual(len(self.fake.requests), 1)
        request = self.fake.requests[0]
        self.assertFalse(request['think'])
        self.assertLessEqual(request['options']['num_predict'], 8192)  # a direct answer may be a full explanation
        self.assertIn('delegate', [schema['function']['name'] for schema in request['tools']])
        self.assertEqual(detail['assistant_budget']['tokens']['used'], 37)
        self.assertEqual(self.ok('GET', '/api/proofs')['jobs'], [])
        self.assertEqual(self.ok('GET', '/api/task')['queue'], [])

    def test_complete_prior_user_turn_is_retained_without_routing_summary(self):
        chat = self.conversation()
        original = 'FIRST HYPOTHESIS\n' + 'preserve this notation exactly\n' * 90 + 'LAST HYPOTHESIS: a is real.'
        self.fake.replies = [text('I have the hypotheses.'), text('Both hypotheses remain available.')]
        self.ask(chat, original)
        self.wait_task()
        self.ask(chat, 'Which hypotheses did I give?')
        self.wait_for(lambda: len(self.fake.requests) == 2, message='second Assistant call')
        self.wait_task()
        second = self.fake.requests[1]
        self.assertIn(original, [message['content'] for message in second['messages'] if message['role'] == 'user'])
        detail = self.detail(chat)
        self.assertEqual(sum(item['role'] == 'user' for item in detail['transcript']), 2)
        self.assertEqual([item['content'] for item in store.load_chat(self.root, chat)['history'] if item['role'] == 'user'],
                         [original, 'Which hypotheses did I give?'])

    def test_assistant_turn_waits_in_the_existing_queue_without_running_a_classifier(self):
        blocker = self.ok('POST', '/api/chats', {'mode': 'critic'})['id']
        slow = ([{'message': {'content': 'working '}, 'done': False}] * 80 + text('Finished.'), 0.01)
        self.fake.replies = [slow, text('A direct answer.')]
        self.ok('POST', f'/api/chats/{blocker}/messages', {'content': 'A slow task.'})
        self.wait_for(lambda: len(self.fake.requests) == 1, message='blocker inference')
        chat = self.conversation()
        self.ask(chat, 'A quick question.')
        snapshot = self.ok('GET', '/api/task')
        self.assertEqual(len(snapshot['queue']), 1)
        self.assertEqual((snapshot['queue'][0]['kind'], snapshot['queue'][0]['target']), ('chat', chat))
        self.assertEqual(len(self.fake.requests), 1)  # the parent cannot bypass the model slot
        self.wait_for(lambda: self.detail(chat).get('assistant_status') == 'complete',
                      message='queued Assistant completion')
        self.assertEqual(len(self.fake.requests), 2)
        self.assertEqual(self.ok('GET', '/api/task')['queue'], [])

    def test_second_user_turn_is_refused_while_the_first_turn_is_running(self):
        chat = self.conversation()
        slow = ([{'message': {'content': 'working '}, 'done': False}] * 80 + text('Finished.'), 0.01)
        self.fake.replies = [slow]
        self.ask(chat, 'First question.')
        self.wait_for(lambda: len(self.fake.requests) == 1, message='Assistant inference')
        status, response, _ = self.request('POST', f'/api/chats/{chat}/route', {'content': 'Second question.'})
        self.assertEqual(status, 409, response)
        self.wait_task()
        self.assertEqual([item['content'] for item in self.detail(chat)['transcript'] if item['role'] == 'user'],
                         ['First question.'])

    def test_proof_worker_returns_to_parent_and_preserves_exact_sources(self):
        chat = self.conversation()
        source = 'Claim: every real x satisfies x=x.\nHypothesis: x is an arbitrary real number.\n'
        (self.root / 'identity.tex').write_text(source)
        query = 'Prove identity.tex. Keep its arbitrary real quantifier exactly.'
        final = 'The candidate uses reflexivity for arbitrary real x. The model review found no issue.'
        self.fake.replies = [delegation('prove', 'Prove the identity in identity.tex.', ['identity.tex']),
                             text(PROOF, eval_count=53), text(json.dumps(APPROVAL), eval_count=31),
                             text(final, eval_count=27)]
        self.ask(chat, query, files=['identity.tex'])
        task = self.wait_task(timeout=30)
        self.assertEqual(task['state'], 'done', task)
        self.assertEqual((task['kind'], task['target']), ('chat', chat))
        detail = self.detail(chat)
        self.assertEqual(detail['assistant_status'], 'complete')
        cards = [item for item in detail['transcript'] if item['role'] == 'route']
        self.assertEqual(len(cards), 1, cards)
        card = cards[0]
        self.assertTrue(card['orchestrated'])
        self.assertEqual((card['mode'], card['status']), ('prove', 'done'))
        proof = self.ok('GET', f'/api/proofs/{card["job_id"]}')
        self.assertEqual((proof['status'], proof['answer']), ('candidate_complete', PROOF))
        self.assertIn(query, proof['goal'])
        self.assertEqual(self.ok('GET', f'/api/proofs/{card["job_id"]}/sources')['sources'][0]['content'], source)
        parent_final = self.fake.requests[-1]
        result = json.loads(next(message['content'] for message in reversed(parent_final['messages'])
                                 if message['role'] == 'tool'))
        self.assertEqual(result['result']['status'], 'candidate_complete')
        self.assertIn(PROOF, json.dumps(result))
        self.assertEqual(next(item['content'] for item in reversed(detail['transcript'])
                              if item['role'] == 'assistant'), final)
        self.assertEqual(detail['assistant_budget']['tokens']['used'], 8 + 53 + 31 + 27)
        self.assertEqual(self.ok('GET', '/api/task')['queue'], [])

    def test_poincare_proof_promotes_shared_budget_and_continues_after_full_cutoff(self):
        self.hub.args.proof_solve_tokens = 32768
        self.hub.args.proof_verify_tokens = 16384
        chat = self.conversation()
        cutoff = text('Unfinished written derivation.', eval_count=32768)
        cutoff[-1]['done_reason'] = 'length'
        self.fake.replies = [delegation('prove', 'Prove the exact original statement.', effort='poincare'),
                             cutoff, text(PROOF, eval_count=20000),
                             text(json.dumps(APPROVAL), eval_count=8000), text('The written proof was reviewed.')]
        original = 'Let G=-d_x^2-x^2*d_y^2 on (-1,1) x S^1. Show non-observability for T<a^2/2.'
        self.ask(chat, original)
        self.wait_task(timeout=30)
        saved = store.load_chat(self.root, chat)['assistant_state']
        self.assertEqual(saved['status'], 'complete', saved['warnings'])
        self.assertEqual(saved['budget']['max_tokens'], 200000)
        self.assertEqual(saved['budget']['max_input_tokens'], 800000)
        self.assertEqual(saved['budget']['max_seconds'], 7200)
        [child] = saved['children']
        proof = self.ok('GET', f'/api/proofs/{child["job_id"]}')
        state = json.loads((self.root / '.mathagent' / 'proofs' / child['job_id'] / 'state.json').read_text())
        self.assertEqual(proof['status'], 'candidate_complete')
        self.assertEqual(state['rounds_started'], 2)
        self.assertEqual(state['settings']['max_rounds'], 10)
        self.assertEqual(state['settings']['max_tokens'], 200000 - 8 - 2 * 8192)
        self.assertIn(original, state['goal'])
        self.assertIn('ORIGINAL USER REQUEST is authoritative', state['goal'])
        self.assertEqual(saved['budget']['tokens'], 8 + 32768 + 20000 + 8000 + 20)

    def test_explicit_assistant_caps_are_not_raised_by_poincare_worker(self):
        self.hub.args.assistant_tokens = 60000
        self.hub.args.assistant_input_tokens = 240000
        self.hub.args.assistant_seconds = 900
        chat = self.conversation()
        self.fake.replies = [delegation('prove', 'Prove identity.', effort='poincare'),
                             text(PROOF), text(json.dumps(APPROVAL)), text('The proof is complete.')]
        self.ask(chat, 'Prove x=x.')
        self.wait_task(timeout=30)
        saved = store.load_chat(self.root, chat)['assistant_state']
        self.assertEqual(saved['status'], 'complete')
        self.assertEqual((saved['budget']['max_tokens'], saved['budget']['max_input_tokens'], saved['budget']['max_seconds']),
                         (60000, 240000, 900))
        self.assertNotIn('adaptive_limits', saved['budget'])

    def test_mixed_parallel_efforts_receive_proportional_shares(self):
        self.openai_fixture()
        chat = self.conversation()
        tasks = delegation('critic', 'Check one sign.', effort='low')
        tasks[0]['message']['tool_calls'] += delegation('explore', 'Work through the harder argument.', effort='poincare')[0]['message']['tool_calls']
        self.fake.replies = [tasks, text('Sign checked.'), text('Exploration finished.'), text('Both findings are retained.')]
        self.ask(chat, 'Run these two independent tasks.')
        self.wait_task(timeout=30)
        saved = store.load_chat(self.root, chat)['assistant_state']
        self.assertEqual(saved['status'], 'complete')
        low, poincare = saved['children']
        self.assertGreater(poincare['allocation'], 6 * low['allocation'])
        self.assertLessEqual(low['allocation'] + poincare['allocation'], 200000 - 8 - 2 * 8192)

    def test_two_independent_workers_use_parallel_calls_and_one_shared_budget(self):
        self.openai_fixture()
        chat = self.conversation()
        first = delegation('critic', 'Check the sign argument.')
        first[0]['message']['tool_calls'] += delegation('explore', 'Find an alternative short argument.')[0]['message']['tool_calls']
        slow = ([{'message': {'content': 'checking '}, 'done': False}] * 40 + text('Both checks are explicit.'), 0.01)
        self.fake.replies = [first, slow, slow, text('The independent workers agree.')]
        self.ask(chat, 'Please make two independent checks.')
        self.wait_for(lambda: len(self.fake.requests) >= 3, message='two concurrent worker calls')
        # Both worker calls have entered the server before either finishes.
        self.assertEqual(len(self.fake.requests), 3)
        self.assertEqual(self.ok('GET', '/api/task')['queue'], [])
        self.wait_task(timeout=30)
        detail = self.detail(chat)
        self.assertEqual(detail['assistant_status'], 'complete')
        cards = [item for item in detail['transcript'] if item['role'] == 'route']
        # The two workers start concurrently, so either may write its card first.
        self.assertEqual(sorted(card['mode'] for card in cards), ['critic', 'explore'])
        self.assertTrue(all(card['status'] == 'done' for card in cards))
        self.assertEqual(detail['assistant_budget']['tokens']['used'], 8 + 20 + 20 + 20)
        self.assertEqual(len(self.fake.requests), 4)

    def test_resume_keeps_the_finished_parallel_worker_and_only_retries_the_interrupted_one(self):
        self.openai_fixture()
        chat = self.conversation()
        first = delegation('critic', 'Check the positive sign case.')
        first[0]['message']['tool_calls'] += delegation('explore', 'Check the negative sign case.')[0]['message']['tool_calls']
        slow = ([{'message': {'content': 'checking '}, 'done': False}] * 120 + text('Second check done.'), 0.01)
        self.fake.replies = [first, text('First check done.'), slow]
        self.ask(chat, 'Make two independent checks of the real square proof.')
        self.wait_for(lambda: sum(child['status'] == 'complete' for child in
                                 store.load_chat(self.root, chat).get('assistant_state', {}).get('children', [])) == 1,
                      message='one finished parallel worker')
        self.ok('POST', '/api/task/pause')
        self.assertEqual(self.wait_task()['state'], 'paused')
        before = store.load_chat(self.root, chat)['assistant_state']
        completed = next(child for child in before['children'] if child['status'] == 'complete')
        self.fake.replies = [text('Second check resumed.'), text('Both saved checks are available.')]
        self.ok('POST', f'/api/chats/{chat}/assistant/resume', {})
        self.wait_for(lambda: self.detail(chat).get('assistant_status') == 'complete', timeout=30,
                      message='resumed parallel workflow')
        after = store.load_chat(self.root, chat)['assistant_state']
        kept = next(child for child in after['children'] if child['id'] == completed['id'])
        self.assertEqual(kept['result'], completed['result'])
        self.assertEqual(len(self.fake.requests), 5)  # main, two workers, one resumed worker, final
        self.assertEqual(len(after['children']), 2)
        self.assertEqual([card['status'] for card in self.detail(chat)['transcript'] if card['role'] == 'route'],
                         ['done', 'done'])

    def test_output_limit_remains_unfinished_and_resumes_without_duplicate_user_message(self):
        chat = self.conversation()
        cutoff = text('If a is positive, then', eval_count=19)
        cutoff[-1]['done_reason'] = 'length'
        self.fake.replies = [cutoff]
        self.ask(chat, 'Prove a^2 >= 0.')
        self.wait_task()
        detail = self.detail(chat)
        self.assertEqual(detail['assistant_status'], 'paused')
        self.assertEqual(store.load_chat(self.root, chat)['assistant_state']['status'], 'incomplete')
        self.assertEqual(detail['assistant_budget']['tokens']['used'], 19)
        self.fake.replies = [text('The sign cases show a^2 is nonnegative.', eval_count=23)]
        self.ok('POST', f'/api/chats/{chat}/assistant/resume', {})
        self.wait_for(lambda: self.detail(chat).get('assistant_status') == 'complete',
                      message='resumed Assistant answer')
        detail = self.detail(chat)
        self.assertEqual(sum(item['role'] == 'user' for item in detail['transcript']), 1)
        self.assertEqual(detail['assistant_budget']['tokens']['used'], 19 + 23)

    def test_capped_synthesis_resumes_with_a_larger_answer_without_rerunning_the_completed_proof(self):
        chat = self.conversation()
        query = 'Prove x=x for every real x, then explain the completed proof.'
        cutoff = text('The reviewed proof is complete. Its explanation begins with', eval_count=8192)
        cutoff[-1]['done_reason'] = 'length'
        self.fake.replies = [delegation('prove', 'Prove that every real x satisfies x=x.'),
                             text(PROOF, eval_count=53), text(json.dumps(APPROVAL), eval_count=31), cutoff]
        self.ask(chat, query)
        self.wait_task(timeout=30)
        before = self.detail(chat)
        self.assertEqual(before['assistant_status'], 'paused')
        [card] = [item for item in before['transcript'] if item['role'] == 'route']
        proof_id = card['job_id']
        self.assertEqual(self.ok('GET', f'/api/proofs/{proof_id}')['status'], 'candidate_complete')
        self.assertEqual(len(self.fake.requests), 4)
        self.assertGreaterEqual(self.fake.requests[-1]['options']['num_predict'], 8192)
        spent = before['assistant_budget']['tokens']['used']
        self.assertEqual(spent, 8 + 53 + 31 + 8192)
        saved = store.load_chat(self.root, chat)['assistant_state']
        self.assertEqual(saved['children'][0]['status'], 'complete')
        self.assertEqual(saved['last_call']['output_limit'], self.fake.requests[-1]['options']['num_predict'])

        replacement = 'For arbitrary real x, equality is reflexive, so x=x. The saved candidate received a model review.'
        self.fake.replies = [text(replacement, eval_count=61)]
        self.ok('POST', f'/api/chats/{chat}/assistant/resume', {})
        # The checkpoint says complete just before the answer is committed to the transcript.
        self.wait_for(lambda: self.detail(chat).get('assistant_status') == 'complete'
                      and any(item.get('content') == replacement for item in self.detail(chat)['transcript']),
                      timeout=30, message='complete replacement synthesis')
        after = self.detail(chat)
        [resumed_card] = [item for item in after['transcript'] if item['role'] == 'route']
        self.assertEqual(resumed_card['job_id'], proof_id)
        self.assertEqual(len(self.ok('GET', '/api/proofs')['jobs']), 1)
        self.assertEqual(len(self.fake.requests), 5)  # resume runs the main agent only
        request = self.fake.requests[-1]
        self.assertNotIn('tools', request)
        self.assertGreaterEqual(request['options']['num_predict'], 8192)
        self.assertGreater(request['options']['num_predict'], self.fake.requests[-2]['options']['num_predict'])
        self.assertIn(PROOF, json.dumps([message for message in request['messages'] if message['role'] == 'tool']))
        self.assertEqual(sum(item['role'] == 'user' for item in after['transcript']), 1)
        self.assertEqual(after['assistant_budget']['tokens']['used'], spent + 61)
        self.assertEqual(next(item['content'] for item in reversed(after['transcript']) if item['role'] == 'assistant'),
                         replacement)

    def test_running_checkpoint_without_a_live_task_is_recoverable_after_restart(self):
        chat = self.conversation()
        cutoff = text('Unfinished model output.', eval_count=19)
        cutoff[-1]['done_reason'] = 'length'
        self.fake.replies = [cutoff]
        self.ask(chat, 'Keep the original hypotheses while recovering.')
        self.wait_task()
        saved = store.load_chat(self.root, chat)['assistant_state']
        saved['status'] = 'running'  # durable state left by an interrupted GUI process
        saved['limits']['actions'] = 6  # retain a checkpoint made before the default increased
        store.save_assistant(self.root, chat, saved)
        self.assertIsNone(self.hub.active())
        self.assertEqual(self.detail(chat)['assistant_status'], 'paused')
        self.fake.replies = [text('The original hypotheses were recovered.', eval_count=23)]
        self.ok('POST', f'/api/chats/{chat}/assistant/resume', {})
        self.wait_for(lambda: self.detail(chat).get('assistant_status') == 'complete',
                      message='recovered Assistant answer')
        detail = self.detail(chat)
        self.assertEqual(sum(item['role'] == 'user' for item in detail['transcript']), 1)
        self.assertEqual(detail['assistant_budget']['tokens']['used'], 19 + 23)

        self.assertEqual(store.load_chat(self.root, chat)['assistant_state']['limits']['actions'], 6)

    def test_paused_proof_worker_resumes_the_same_saved_job(self):
        chat = self.conversation()
        slow = ([{'message': {'content': 'partial proof '}, 'done': False}] * 120 + text(PROOF), 0.01)
        self.fake.replies = [delegation('prove', 'Prove that every real x satisfies x=x.'), slow]
        self.ask(chat, 'Prove x=x for arbitrary real x with independent review.')
        self.wait_for(lambda: len(self.fake.requests) >= 2, message='proof worker inference')
        self.ok('POST', '/api/task/pause')
        self.assertEqual(self.wait_task()['state'], 'paused')
        detail = self.detail(chat)
        self.assertEqual(detail['assistant_status'], 'paused')
        [before] = [item for item in detail['transcript'] if item['role'] == 'route']
        self.assertTrue(before['job_id'])
        spent = detail['assistant_budget']['tokens']['used']
        self.assertGreaterEqual(spent, self.fake.requests[1]['options']['num_predict'])
        self.fake.replies = [text(PROOF), text(json.dumps(APPROVAL)), text('The saved proof was resumed and reviewed.')]
        self.ok('POST', f'/api/chats/{chat}/assistant/resume', {})
        self.wait_for(lambda: self.detail(chat).get('assistant_status') == 'complete', timeout=30,
                      message='resumed proof workflow')
        detail = self.detail(chat)
        [after] = [item for item in detail['transcript'] if item['role'] == 'route']
        self.assertEqual(after['job_id'], before['job_id'])
        self.assertEqual(after['status'], 'done')
        self.assertEqual(len(self.ok('GET', '/api/proofs')['jobs']), 1)
        self.assertEqual(sum(item['role'] == 'user' for item in detail['transcript']), 1)
        self.assertGreater(detail['assistant_budget']['tokens']['used'], spent)

    def test_disabled_file_types_remain_unreadable_in_main_agent(self):
        chat = self.conversation(online=False)
        (self.root / 'private.py').write_text('PRIVATE_SENTINEL=12345\n')
        self.ok('POST', '/api/settings', {'read': {'py': False}})
        self.fake.replies = [tool('read_file', {'path': 'private.py'}), text('That file type is unavailable.')]
        self.ask(chat, 'Can you read private.py?')
        self.wait_task()
        result = next(message['content'] for message in self.fake.requests[-1]['messages'] if message['role'] == 'tool')
        self.assertIn('not allowed', result)
        self.assertNotIn('PRIVATE_SENTINEL', result)
        schemas = self.fake.requests[0]['tools']
        self.assertNotIn('write_file', [item['function']['name'] for item in schemas])
        self.assertNotIn('run_python', [item['function']['name'] for item in schemas])

    def test_explicit_pin_is_readable_without_unlocking_other_disabled_files(self):
        chat = self.conversation(online=False)
        (self.root / 'source.py').write_text('EXACT_PINNED_HYPOTHESIS = "a is real"\n')
        (self.root / 'unrelated.py').write_text('UNRELATED_PRIVATE_SENTINEL = 12345\n')
        self.ok('POST', '/api/settings', {'read': {'tex': False, 'pdf': False, 'py': False, 'text': False}})
        self.fake.replies = [tool('read_file', {'path': 'source.py'}),
                             tool('read_file', {'path': 'unrelated.py'}), text('The pinned hypothesis is that a is real.')]
        self.ask(chat, 'Read my explicitly pinned source.py.', files=['source.py'])
        self.wait_task()
        detail = self.detail(chat)
        self.assertEqual(detail['assistant_status'], 'complete')
        results = [message['content'] for message in self.fake.requests[-1]['messages'] if message['role'] == 'tool']
        self.assertEqual(len(results), 2)
        self.assertIn('EXACT_PINNED_HYPOTHESIS', results[0])
        self.assertIn('not allowed', results[1])
        self.assertNotIn('UNRELATED_PRIVATE_SENTINEL', '\n'.join(results))
        self.assertIn('read_file', [item['function']['name'] for item in self.fake.requests[0]['tools']])

    def test_offline_choice_is_inherited_by_the_delegated_literature_worker(self):
        chat = self.conversation(online=False)
        scope = {'topic': 'Reflexivity of equality', 'subfield': 'Logic', 'msc': ['03B10'], 'intent': 'learn',
                 'assumptions': 'Cached sources only.', 'queries': ['reflexivity of equality'], 'seeds': []}
        self.fake.replies = [delegation('literature', 'Survey reflexivity using local cached sources.'),
                             text(json.dumps(scope)),
                             text('The literature worker found no cached source on reflexivity.')]
        self.ask(chat, 'Survey reflexivity using only cached sources.')
        self.wait_task(timeout=30)
        detail = self.detail(chat)
        self.assertEqual(detail['assistant_status'], 'complete')
        [card] = [item for item in detail['transcript'] if item['role'] == 'route']
        research = self.ok('GET', f'/api/research/{card["job_id"]}')
        self.assertFalse(research['settings']['online'])
        self.assertEqual(research['pipeline'], 'reading-list-v1')  # the verified reading list
        state = store.research_state(self.root, card['job_id'])
        self.assertTrue(state['sweep_log'])
        self.assertTrue(all(entry.get('count') in (None, 0) for entry in state['sweep_log']), state['sweep_log'])

    def test_reference_request_is_delegated_to_a_literature_check(self):
        from mathagent import refcheck
        from tests.test_gui_features import NOT_FOUND, RECALL
        compact = patch.dict(refcheck.EFFORTS, {'low': {**refcheck.EFFORTS['low'], 'candidates': 2, 'discover': False}})
        compact.start()
        self.addCleanup(compact.stop)
        chat = self.conversation(online=False)
        self.fake.replies = [delegation('check', 'Find a reference for H^2 regularity of the Neumann problem.'),
                             text(json.dumps(RECALL)), text('Nothing to search.'), text(json.dumps(NOT_FOUND)),
                             text('Brezis, Theorem 9.26, is a likely source but the check could not confirm it.')]
        self.ask(chat, 'Find the exact reference for H^2 regularity for the Neumann problem.')
        self.wait_task(timeout=30)
        detail = self.detail(chat)
        self.assertEqual(detail['assistant_status'], 'complete')
        delegate = next(item for item in self.fake.requests[0]['tools'] if item['function']['name'] == 'delegate')
        self.assertIn('check', delegate['function']['parameters']['properties']['mode']['enum'])
        self.assertIn('Use check to find a useful standard reference', self.fake.requests[0]['messages'][0]['content'])
        [card] = [item for item in detail['transcript'] if item['role'] == 'route']
        self.assertEqual((card['mode'], card['status'], card['orchestrated']), ('check', 'done', True))
        self.assertTrue((self.root / '.mathagent' / 'checks' / card['job_id'] / 'report.md').exists())
        self.assertEqual(self.ok('GET', '/api/research')['jobs'], [])  # not a reading list
        # The check sees only the objective: no conversation and no workspace files.
        recall = self.fake.requests[1]
        self.assertEqual(recall['format']['required'], ['statement', 'keywords', 'candidates'])
        self.assertNotIn('Where is H^2 regularity', json.dumps(recall['messages']))
        saved = store.load_chat(self.root, chat)['assistant_state']
        [result] = [json.loads(m['content']) for m in saved['messages'] if m['role'] == 'tool']
        self.assertEqual(result['mode'], 'check')
        self.assertIn('No reference could be confirmed', result['result']['answer'])
        self.assertEqual(result['result']['references']['not_confirmed'][0]['level'], 'not_found')
        self.assertIn('Online search was off', result['result']['warnings'][0])
        self.assertEqual(next(item['content'] for item in reversed(detail['transcript']) if item['role'] == 'assistant'),
                         'Brezis, Theorem 9.26, is a likely source but the check could not confirm it.')

    def test_check_recovery_warning_and_unexecuted_followup_reach_saved_chat(self):
        from mathagent import refcheck
        from tests.test_gui_features import NOT_FOUND, RECALL
        with patch.dict(refcheck.EFFORTS, {'low': {**refcheck.EFFORTS['low'], 'candidates': 2, 'discover': False}}):
            chat = self.conversation(online=False)
            cutoff = text('{"candidates":[{"year":' + ' ' * 1000)
            cutoff[-1]['done_reason'] = 'length'
            answer = 'The proposed reference remains unconfirmed; online search was disabled.'
            self.fake.replies = [delegation('check', 'Verify the reference.'), cutoff,
                                 text(json.dumps(RECALL)), text('Nothing to search.'), text(json.dumps(NOT_FOUND)),
                                 text('Let me verify the source.'), text(answer)]
            self.ask(chat, 'Verify this citation.')
            self.wait_task(timeout=30)
        saved = store.load_chat(self.root, chat)['assistant_state']
        self.assertEqual(saved['status'], 'complete')
        self.assertEqual(saved['answer'], answer)
        self.assertTrue(saved['continuation_final_retried'])
        self.assertNotIn('format', self.fake.requests[2])
        [child] = saved['children']
        self.assertTrue(any('constrained decoding' in w for w in child['result']['warnings']))
        self.assertEqual(len(list((self.root / '.mathagent' / 'checks').glob('*/state.json'))), 1)

    def test_standard_book_lookup_finishes_with_qualified_answer_after_one_worker(self):
        from tests.test_gui_features import RECALL
        chat = self.conversation(online=False)
        # Even an over-specific delegated restatement cannot strengthen the human request.
        self.fake.replies = [delegation('check', 'Find the exact theorem number for this result.'),
                             text(json.dumps(RECALL))]
        self.ask(chat, 'Find a standard reference for Neumann regularity without an exact theorem number.')
        self.wait_task(timeout=30)
        saved = store.load_chat(self.root, chat)['assistant_state']
        self.assertEqual(saved['status'], 'complete')
        self.assertEqual(saved['actions'], 1)
        [child] = saved['children']
        self.assertEqual(saved['answer'], child['result']['answer'])
        self.assertEqual(saved['main_calls'], 1)
        self.assertEqual(saved['answer_source'], {'kind': 'worker_result', 'job_id': child['job_id']})
        self.assertEqual(child['result']['disposition'], 'answer_with_qualification')
        self.assertIn('Brezis', saved['answer'])
        self.assertIn('unconfirmed', saved['answer'].lower())
        self.assertNotIn('Springer', saved['answer'])
        self.assertNotIn('edition', saved['answer'].lower())
        state = json.loads((self.root / '.mathagent' / 'checks' / child['job_id'] / 'state.json').read_text())
        self.assertEqual(state['check']['lookup'], 'standard')
        self.assertEqual(state['settings']['max_requests'], 4)
        self.assertEqual(len(state['candidates']), 1)
        self.assertEqual([e['tool'] for e in state['evidence']], ['search_papers'])
        self.assertNotIn('tools', self.fake.requests[-1])
        self.assertEqual(len(self.fake.requests), 2)

    def test_critic_worker_can_check_references_like_a_critique_chat(self):
        chat = self.conversation(online=True)
        self.fake.replies = [delegation('critic', 'Check that x=x for every real x.'), text('Reflexivity holds.'),
                             text('Equality is reflexive.')]
        self.ask(chat, 'Is x=x for every real x?')
        self.wait_task(timeout=30)
        self.assertEqual(self.detail(chat)['assistant_status'], 'complete')
        names = lambda request: [item['function']['name'] for item in request.get('tools', [])]
        self.assertNotIn('check_reference', names(self.fake.requests[0]))  # the main agent delegates a check
        self.assertIn('check_reference', names(self.fake.requests[1]))
        offline = self.conversation(online=False)
        self.fake.replies = [delegation('critic', 'Check that y=y for every real y.'), text('Reflexivity holds.'),
                             text('Equality is reflexive.')]
        self.ask(offline, 'Is y=y for every real y?')
        self.wait_task(timeout=30)
        self.assertNotIn('check_reference', names(self.fake.requests[-2]))

    def test_large_previous_report_is_reference_context_not_an_oversized_literature_goal(self):
        self.hub.args.ctx = 40960  # the exact conversation fits when supplied only once
        chat = self.conversation(online=False)
        source = 'Hypotheses: a is real, c > 0, and T > T0.\nKeep every quantifier unchanged.\n'
        (self.root / 'hypotheses.tex').write_text(source)
        previous_report = 'PREVIOUS_REPORT_START\n' + 'Exact prior report material.\n' * 900 + 'PREVIOUS_REPORT_END'
        self.assertGreater(len(previous_report), 20000)
        prior_query = 'Use precisely these hypotheses:\n' + source
        store.commit_turn(self.root, chat,
                          [{'role': 'user', 'content': prior_query},
                           {'role': 'assistant', 'content': previous_report}], [], [])
        goal = 'Find literature support under the exact pinned hypotheses.'
        scope = {'topic': 'Support under pinned hypotheses', 'subfield': 'Analysis', 'msc': [], 'intent': 'learn',
                 'assumptions': 'All pinned hypotheses kept.', 'queries': ['reflexivity of equality'], 'seeds': []}
        self.fake.replies = [delegation('literature', goal, ['hypotheses.tex']), text(json.dumps(scope)),
                             text('The source limitation remains explicit.')]
        self.ask(chat, 'Find support for the previous report without weakening any hypothesis.', ['hypotheses.tex'])
        self.wait_task(timeout=30)
        detail = self.detail(chat)
        self.assertEqual(detail['assistant_status'], 'complete', detail)
        [card] = [item for item in detail['transcript'] if item['role'] == 'route']
        self.assertEqual(card['status'], 'done', card)
        state = store.research_state(self.root, card['job_id'])
        parent = store.load_chat(self.root, chat)['assistant_state']
        [child] = parent['children']
        self.assertEqual(state['goal'], goal)
        self.assertLess(len(state['goal']), 20000)
        self.assertEqual(state['reference_context'], child['context'])
        self.assertIn(previous_report, [message['content'] for message in state['reference_context']['history']])
        self.assertIn(source, [message['content'].split('Use precisely these hypotheses:\n', 1)[-1]
                               for message in state['reference_context']['history']])
        [pinned] = state['sources']
        self.assertEqual(pinned['content'], source)
        self.assertEqual(pinned['sha256'], hashlib.sha256(source.encode()).hexdigest())
        material = '\n'.join(message.get('content', '') for message in self.fake.requests[1]['messages'])
        self.assertIn('PREVIOUS_REPORT_START', material)
        self.assertIn('PREVIOUS_REPORT_END', material)
        self.assertIn(json.dumps(previous_report)[1:-1], material)
        self.assertEqual(material.count('PREVIOUS_REPORT_START'), 1)
        self.assertEqual(material.count('PREVIOUS_REPORT_END'), 1)
        self.assertIn('Hypotheses: a is real, c > 0, and T > T0.', material)
        self.assertIn('Keep every quantifier unchanged.', material)
        self.assertNotIn('Unexpected extra call.', json.dumps(detail))

    def test_permanent_worker_failure_blocks_paraphrase_and_effort_changes_but_allows_another_mode(self):
        chat = self.conversation(online=False)
        error = WorkerInputError('The exact inherited context cannot fit this worker.',
                                 code='reference_context_too_large',
                                 details={'context_chars': 75000, 'limit': 20000})
        repeated = delegation('literature', 'Try the same source survey with a differently phrased objective.', effort='high')
        repeated[0]['message']['tool_calls'] += delegation('critic', 'Check the independent algebraic inference.')[0]['message']['tool_calls']
        self.fake.replies = [delegation('literature', 'Survey the sources for this claim.'), repeated,
                             text('The independent inference remains unproved.'),
                             text('The literature task is blocked; the separate critique identified its own uncertainty.')]
        with patch('mathagent.gui.hub.ResearchRunner.start', side_effect=error) as startup:
            self.ask(chat, 'Survey sources and independently critique the argument.')
            self.wait_task(timeout=30)
        self.assertEqual(startup.call_count, 1)
        detail = self.detail(chat)
        self.assertEqual(detail['assistant_status'], 'complete')
        saved = store.load_chat(self.root, chat)['assistant_state']
        self.assertEqual([child['action']['mode'] for child in saved['children']], ['literature', 'critic'])
        self.assertEqual(len(saved['blockers']), 1)
        self.assertEqual(saved['blockers'][0]['mode'], 'literature')
        self.assertEqual(saved['blockers'][0]['error'], error.as_dict())
        self.assertTrue(saved['blockers'][0]['context_sha256'])
        tool_results = [json.loads(message['content']) for message in saved['messages'] if message['role'] == 'tool']
        self.assertTrue(any(result.get('error', {}).get('retryable') is False for result in tool_results
                            if isinstance(result.get('error'), dict)))
        self.assertEqual(len(self.fake.requests), 4)
        self.assertEqual(self.ok('GET', '/api/research')['jobs'], [])

    def test_permanent_failure_blocker_survives_pause_and_saved_resume(self):
        chat = self.conversation(online=False)
        error = WorkerInputError('The report requires more context than this worker supports.',
                                 code='reference_context_too_large', details={'input_tokens': 50000})
        slow = ([{'message': {'content': 'Preparing the next step. '}, 'done': False}] * 100 + text('Unfinished.'), 0.01)
        self.fake.replies = [delegation('literature', 'Check the supplied source report.'), slow]
        with patch('mathagent.gui.hub.ResearchRunner.start', side_effect=error) as startup:
            self.ask(chat, 'Check the source report, retaining the exact hypothesis c > 0.')
            self.wait_for(lambda: len(self.fake.requests) == 2, message='controller after permanent failure')
            self.ok('POST', '/api/task/pause')
            self.assertEqual(self.wait_task()['state'], 'paused')
            before = json.loads(json.dumps(store.load_chat(self.root, chat)['assistant_state']))
            self.assertEqual(len(before['blockers']), 1)
            repeated = delegation('literature', 'Give the bibliography audit another name and more effort.', effort='xhigh')
            repeated[0]['message']['tool_calls'] += delegation('critic', 'Check the unrelated inference under c > 0.')[0]['message']['tool_calls']
            self.fake.replies = [repeated, text('The independent check is available.'),
                                 text('The bibliography audit remains blocked; the independent check was performed.')]
            self.ok('POST', f'/api/chats/{chat}/assistant/resume', {})
            self.wait_for(lambda: self.detail(chat).get('assistant_status') == 'complete', timeout=30,
                          message='resumed turn with durable blocker')
        after = store.load_chat(self.root, chat)['assistant_state']
        self.assertEqual(startup.call_count, 1)
        self.assertEqual(after['blockers'], before['blockers'])
        self.assertEqual([child['action']['mode'] for child in after['children']], ['literature', 'critic'])
        self.assertEqual(sum(item['role'] == 'user' for item in self.detail(chat)['transcript']), 1)
