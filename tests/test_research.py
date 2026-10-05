"""Research budgets, provenance and durable workflow behavior without network/model calls."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mathagent.agent import Agent
from mathagent.literature import LiteratureTools
from mathagent.research import ResearchRunner, _reference_prompt, _reference_snapshot
from mathagent.tools import Workspace


DOC = 'doc-' + 'a' * 24


def response(text='', calls=(), *, complete=True, count=20, input_count=100):
    return [{'message': {'content': text, 'tool_calls': list(calls)}, 'done': True,
             'done_reason': 'stop' if complete else 'length', 'eval_count': count,
             'prompt_eval_count': input_count}]


def tool(name, **args):
    return {'function': {'name': name, 'arguments': args}}


class FakeClient:
    host = 'http://fake.invalid'
    timeout = 60

    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def stream(self, payload):
        self.requests.append(copy.deepcopy(payload))
        for event in next(self.replies):
            if isinstance(event, BaseException):
                raise event
            yield event


class FakeLiterature:
    online = True
    deadline = None
    on_budget_change = None
    max_requests = 12
    max_chars = 30000

    def __init__(self):
        self.stats = {'requests': 0, 'returned_chars': 0}
        self.calls = []

    def reset_budget(self, **kwargs):
        self.max_requests = kwargs['max_requests']
        self.max_chars = kwargs['max_chars']
        self.stats = {'requests': 0, 'returned_chars': 0}

    def snapshot(self):
        return {'online': self.online, 'stats': self.stats.copy(),
                'max_requests': self.max_requests, 'max_chars': self.max_chars}

    def restore(self, value):
        self.online = self.online and value['online']
        self.stats = value['stats'].copy()

    def schemas(self):
        return [{'type': 'function', 'function': {'name': n, 'parameters': {'type': 'object'}}}
                for n in ('search_papers', 'open_paper', 'read_paper', 'search_paper')]

    def execute(self, name, args):
        self.calls.append((name, args))
        if name == 'search_papers':
            self.stats['requests'] += 1
            if self.on_budget_change:
                self.on_budget_change()
            result = {'results': [{'title': 'An example result', 'url': 'https://example.org/paper'}]}
        elif name == 'open_paper':
            result = {'document_id': DOC, 'title': 'An example result', 'source_url': 'https://example.org/paper'}
        else:
            result = {'document_id': DOC, 'source_url': 'https://example.org/paper',
                      'start_line': 1, 'end_line': 2, 'passage': '1: Assumption A.\n2: Under A, conclusion B.'}
        data = json.dumps(result)
        self.stats['returned_chars'] += len(data)
        if self.on_budget_change:
            self.on_budget_change()
        return data


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def runner(self, replies, literature=None, **kwargs):
        workspace = Workspace(self.root)
        workspace.literature = literature
        self.client = FakeClient(replies)
        return ResearchRunner(Agent(self.client, workspace, ctx=kwargs.get('ctx', 16384), predict=kwargs.get('predict', 4096)))

    def test_reference_prompt_reuses_only_exact_duplicate_history(self):
        history = [{'role': 'user', 'content': 'All hypotheses and quantifiers.'},
                   {'role': 'tool', 'content': 'Exact passage.', 'tool_call_id': 'native-id'}]
        context = {'query': 'Check this.', 'history': history,
                   'messages': history + [{'role': 'user', 'content': 'Check this.'}], 'files': []}
        original, _, digest = _reference_snapshot(context)
        rendered = json.loads(_reference_prompt(context))
        self.assertEqual(rendered['messages'], context['messages'])
        self.assertEqual(rendered['history'], {'same_as_messages_prefix': 2})
        self.assertEqual(rendered['messages'][:2], history)
        self.assertEqual(_reference_snapshot(context), _reference_snapshot(original))
        self.assertEqual(_reference_snapshot(context)[2], digest)
        context['history'] = [{'role': 'user', 'content': 'Additional exact hypothesis.'}]
        self.assertEqual(json.loads(_reference_prompt(context)), context)

    def test_real_phases_and_exact_source_provenance(self):
        literature = FakeLiterature()
        citation = f'[{DOC}:L1-L2]'
        draft = '# Survey\nUnder assumption A the source states B. ' + citation
        replies = [response('Plan scope and assumptions.'),
                   response(calls=[tool('search_papers', query='public topic'), tool('open_paper', identifier_or_url='https://example.org/paper'), tool('read_paper', document_id=DOC)]),
                   response('Evidence sufficient; hypothesis A matters.'), response(draft),
                   response('The source states B under A; retain that assumption.'), response(draft)]
        runner = self.runner(replies, literature)
        result = runner.start('Review the literature on a public topic')
        self.assertEqual(result['status'], 'reviewed')
        self.assertIn('Encountered references', result['report'])
        self.assertIn('https://example.org/paper', result['report'])
        self.assertEqual([c['role'] for c in runner.state['calls']], ['plan', 'investigate', 'investigate', 'draft', 'review', 'revise'])
        self.assertEqual(len(runner.state['evidence']), 3)
        self.assertEqual(runner.state['tokens_charged'], 120)
        self.assertEqual(runner.state['input_tokens_charged'], 600)
        self.assertIsNone(literature.on_budget_change)
        self.assertIsNone(literature.deadline)
        for record in runner.state['calls']:
            self.assertTrue((runner.directory / 'artifacts' / record['request']).exists())
        self.assertEqual(ResearchRunner.inspect(self.root, result['id'])['report'], result['report'])
        self.assertEqual(ResearchRunner.list(self.root)[0]['id'], result['id'])
        review_request = self.client.requests[4]
        self.assertIn('Under A, conclusion B.', review_request['messages'][1]['content'])
        self.assertNotIn('Unverified working note:', review_request['messages'][1]['content'])

    def test_fabricated_or_unread_citations_cannot_be_reviewed(self):
        literature = FakeLiterature()
        bad = f'# Survey\nClaim [{DOC}:L1-L20] [doc-invented:L1-L2] [invented](https://unseen.invalid/paper)'
        runner = self.runner([response('Plan'), response(calls=[tool('read_paper', document_id=DOC)]),
                              response('Enough'), response(bad), response('Looks good.'), response(bad)], literature)
        result = runner.start('Review a topic')
        self.assertEqual(result['status'], 'partial')
        self.assertTrue(any('lines not read' in x for x in runner.state['citation_issues']))
        self.assertTrue(any('Unknown source' in x for x in runner.state['citation_issues']))
        self.assertTrue(any('URL was not encountered' in x for x in runner.state['citation_issues']))

    def test_referee_uses_pinned_manuscript_and_offline_tools(self):
        original = 'Theorem. Assume A.\nProof. Then B.\n'
        (self.root / 'manuscript.tex').write_text(original)
        literature = LiteratureTools(self.root, online=False)
        draft = '# Referee draft\nUnresolved concern at [M1:L1-L2]. Offline literature coverage only.'
        runner = self.runner([response('Read the theorem and proof.'), response(calls=[tool('read_manuscript', source_id='M1', start_line=1, end_line=2), tool('search_papers', query='public topic')]),
                              response('Need a justification.'), response(draft), response('Concern is uncertainty, not a demonstrated error.'), response(draft)], literature)
        with patch.object(literature, '_fetch', side_effect=AssertionError('Offline fetch forbidden')):
            result = runner.start('Referee manuscript.tex', kind='referee', max_requests=0)
        self.assertEqual(result['status'], 'reviewed')
        self.assertEqual(runner.state['sources'][0]['content'], original)
        self.assertIn('Offline mode', runner.state['evidence'][1]['result'])
        self.assertEqual(runner.state['manuscript_ranges']['M1'], [[1, 2]])
        self.assertIn('Mathematical referee report', runner.state['skill'])
        self.assertEqual(literature.stats['requests'], 0)
        self.assertNotIn(original, json.dumps(runner.state['evidence'][1]['arguments']))

    def test_resume_retains_budget_and_original_source(self):
        (self.root / 'paper.md').write_text('Pinned text.\n')
        runner = self.runner([response('Plan'), [{'message': {'content': 'Partial note'}}, KeyboardInterrupt()]])
        result = runner.start('Review paper.md', max_rounds=2)
        self.assertEqual(result['status'], 'paused')
        charged = runner.state['tokens_charged']
        self.assertGreaterEqual(charged, 2600)
        (self.root / 'paper.md').write_text('Changed source.\n')
        runner2 = self.runner([response('No new evidence.'), response('# Partial report\nCoverage is limited.'), response('No literature checked.'), response('# Partial report\nCoverage is limited.')])
        resumed = runner2.resume(result['id'])
        self.assertIn(resumed['status'], {'reviewed', 'partial'})
        self.assertGreater(runner2.state['tokens_charged'], charged)
        self.assertEqual(runner2.state['sources'][0]['content'], 'Pinned text.\n')
        self.assertEqual(runner2.state['calls'][1]['status'], 'abandoned')
        self.assertTrue(any(not n['complete'] and n['text'] == 'Partial note' for n in runner2.state['notes']))

    def test_truncated_draft_is_partial_and_unexecuted_tools_stay_unexecuted(self):
        literature = FakeLiterature()
        runner = self.runner([response('Plan'), response('Incomplete investigation', [tool('search_papers', query='x')], complete=False),
                              response('# Partial', complete=False), response('Needs more work.'), response('# Still partial', complete=False)], literature)
        result = runner.start('Review a topic')
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(literature.calls, [])
        self.assertFalse(runner.state['draft_complete'])

    def test_output_budget_reserves_report_and_stops(self):
        # The report request has a 320-token cap after reserving review budget.
        runner = self.runner([response('Plan', count=500), response('# Partial report', count=300), response('Unfinished.', count=100)], predict=512)
        result = runner.start('Review a topic', max_tokens=1024, max_input_tokens=10000)
        self.assertEqual(runner.state['rounds_started'], 0)
        self.assertIn(result['status'], {'partial', 'reviewed', 'budget_exhausted'})
        self.assertLessEqual(runner.state['tokens_charged'], 1024)
        self.assertIn('Report draft', result['report'])

    def test_input_budget_prevents_dispatch_and_reports_partial(self):
        runner = self.runner([])
        result = runner.start('A fairly broad review topic', max_input_tokens=2048)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(self.client.requests, [])
        self.assertIn('Input-token budget', result['report'])

    def test_failure_keeps_exact_evidence_and_report(self):
        literature = FakeLiterature()
        runner = self.runner([response('Plan'), response(calls=[tool('read_paper', document_id=DOC)]), [RuntimeError('transport stopped')]], literature)
        result = runner.start('Review a topic')
        self.assertEqual(result['status'], 'error')
        self.assertEqual(len(runner.state['evidence']), 1)
        self.assertIn('Under A, conclusion B.', runner.state['evidence'][0]['result'])
        self.assertTrue(Path(result['report_path']).exists())
        self.assertIn('transport stopped', result['report'])

    def test_offline_resume_cannot_restore_online_permission(self):
        literature = FakeLiterature()
        runner = self.runner([response('Plan'), [KeyboardInterrupt()]], literature)
        first = runner.start('Review a topic')
        offline = FakeLiterature()
        offline.online = False
        second = self.runner([response('No new evidence'), response('# Report\nNo live literature searches performed.'), response('Limited coverage.'), response('# Report\nNo live literature searches performed.')], offline)
        result = second.resume(first['id'])
        self.assertFalse(offline.online)
        self.assertIn('Online access in this session: False', result['report'])

    def test_invalid_source_hash_and_path_are_rejected(self):
        runner = self.runner([response('Plan'), [KeyboardInterrupt()]])
        result = runner.start('Review a topic')
        with self.assertRaises(ValueError):
            runner.resume('../escape')
        statefile = Path(result['directory']) / 'state.json'
        state = json.loads(statefile.read_text())
        state['skill'] += '\nchanged'
        statefile.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, 'skill digest'):
            runner.resume(result['id'])

    def test_interrupted_tool_reservation_durable_and_not_replayed(self):
        literature = FakeLiterature()
        runner = self.runner([response('Plan'), response(calls=[tool('search_papers', query='public topic')])], literature)
        original = literature.execute
        def interrupted(name, args):
            original(name, args)
            raise KeyboardInterrupt()
        literature.execute = interrupted
        result = runner.start('Review a topic')
        self.assertEqual(result['status'], 'paused')
        self.assertEqual(runner.state['literature_snapshot']['stats']['requests'], 1)
        second = self.runner([response('No more tools.'), response('# Report\nIncomplete coverage.'), response('Interrupted evidence missing.'), response('# Report\nIncomplete coverage.')], FakeLiterature())
        resumed = second.resume(result['id'])
        self.assertEqual(second.literature.calls, [])
        self.assertEqual(second.literature.stats['requests'], 1)
        self.assertIn('interrupted tool request was not replayed', resumed['report'])

    def test_review_losing_cited_passage_for_context_stays_partial(self):
        literature = FakeLiterature()
        original = literature.execute
        count = [0]
        def large(name, args):
            result = json.loads(original(name, args))
            count[0] += 1
            result['document_id'] = DOC if count[0] == 1 else 'doc-' + 'b' * 24
            result['passage'] = 'An exact long mathematical statement. ' * 145
            return json.dumps(result)
        literature.execute = large
        draft = f'# Survey\nA claim [{DOC}:L1-L2].'
        runner = self.runner([response('Plan'), response(calls=[tool('read_paper', document_id=DOC), tool('read_paper', document_id='doc-' + 'b' * 24)]),
                              response('Enough evidence.'), response(draft), response('Approved.'), response(draft)], literature, ctx=4096)
        result = runner.start('Review a topic')
        self.assertEqual(result['status'], 'partial')
        self.assertTrue(runner.state.get('review_context_issues'))
        self.assertIn('could not receive every cited passage', result['report'])

    def test_local_pdf_is_imported_locally_and_pinned_as_manuscript(self):
        literature = FakeLiterature()
        literature.import_local = lambda path: {'document_id': DOC}
        literature._document = lambda document_id: ({}, ['Theorem. Assume A.', 'Proof. Then B.'])
        draft = '# Referee\nAn unresolved issue [M1:L1-L2].'
        runner = self.runner([response('Plan'), response(calls=[tool('read_manuscript', source_id='M1', start_line=1, end_line=2)]),
                              response('Read'), response(draft), response('Concern retained.'), response(draft)], literature)
        result = runner.start('Referee manuscript.pdf', kind='referee')
        self.assertEqual(result['status'], 'reviewed')
        self.assertEqual(runner.state['sources'][0]['path'], 'manuscript.pdf')
        self.assertIn('Theorem. Assume A.', runner.state['sources'][0]['content'])
        self.assertEqual(literature.calls, [])

    def test_resume_resets_unrelated_backend_counters(self):
        literature = FakeLiterature()
        runner = self.runner([response('Plan'), [KeyboardInterrupt()]], literature)
        first = runner.start('Review a topic')
        literature.stats = {'requests': 99, 'returned_chars': 99999}
        second = self.runner([response('No more research'), response('# Report'), response('Limited scope.'), response('# Report')], literature)
        second.resume(first['id'])
        self.assertEqual(literature.stats, {'requests': 0, 'returned_chars': 0})

    def test_source_references_use_only_actual_read_ranges(self):
        runner = self.runner([response('Plan'), [KeyboardInterrupt()]])
        runner.start('Review a topic')
        runner.state['evidence'] = [{'result': json.dumps({'document_id': DOC, 'source_url': 'https://example.org/paper', 'matches': [{'line': 5, 'passage': 'Exact full line', 'partial_line': False}, {'line': 10, 'passage': 'part', 'partial_line': True}]})}]
        self.assertEqual(runner._citation_issues(f'Claim [{DOC}:L5]'), [])
        self.assertTrue(runner._citation_issues(f'Claim [{DOC}:L10]'))
        runner.state['evidence'] = [{'result': json.dumps({'document_id': DOC, 'start_line': 1, 'end_line': 20, 'truncated': True, 'passage': '1: Partial...'})}]
        self.assertTrue(runner._citation_issues(f'Claim [{DOC}:L1-L20]'))


if __name__ == '__main__':
    unittest.main()
