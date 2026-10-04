"""Native Assistant browsing and complete reference pagination without HTTP/model access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mathagent.agent import Agent
from mathagent.literature import LiteratureTools
from mathagent.orchestrator import Orchestrator
from mathagent.tools import Workspace
from mathagent.web import WebTools


URL = 'https://example.org/paper-with-bibliography'


def bibliography(count):
    entries = []
    for number in range(1, count + 1):
        link = '' if number % 3 == 0 else ' <a href="https://example.org/shared-source">source</a>'
        entries.append(f'<li id="ref-{number}">[{number}] Author {number}. Exact reference title {number}.{link}</li>')
    return ('<!doctype html><html><head><title>A source paper</title></head><body>'
            '<h1>A source paper</h1><p>The supplied hypothesis is c &gt; 0.</p>'
            '<h2>References</h2><ol class="references">' + ''.join(entries) + '</ol></body></html>').encode()


def native(name, arguments):
    return [{'message': {'tool_calls': [{'function': {'name': name, 'arguments': arguments}}]},
             'done': True, 'done_reason': 'tool_calls', 'eval_count': 20, 'prompt_eval_count': 100}]


def answer(content):
    return [{'message': {'content': content}, 'done': True, 'done_reason': 'stop',
             'eval_count': 20, 'prompt_eval_count': 100}]


class ScriptedClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def stream(self, payload):
        self.requests.append(copy.deepcopy(payload))
        if not self.replies:
            raise AssertionError('Unexpected model call in the browsing fixture')
        reply = self.replies.pop(0)
        yield from reply(payload) if callable(reply) else reply


def source_result(payload):
    message = next(item for item in reversed(payload['messages']) if item['role'] == 'tool')
    return json.loads(message['content'])


class AssistantWebTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def runner(self, literature, replies):
        client = ScriptedClient(replies)
        worker = Mock(side_effect=AssertionError('Direct browsing must not start a literature worker'))
        workspace = Workspace(self.root, literature=literature)
        runner = Orchestrator(Agent(client, workspace, ctx=32768), worker)
        return runner, client, worker

    def test_native_assistant_opens_url_reads_references_and_answers_without_a_worker(self):
        literature = LiteratureTools(self.root, online=True)
        opened = []

        def references(payload):
            page = source_result(payload)
            opened.append(page)
            return native('read_references', {'page_id': page['page_id'], 'start': 1, 'limit': 20})

        def synthesis(payload):
            refs = source_result(payload)
            self.assertEqual(refs['total'], 6)
            self.assertEqual(len(refs['references']), 6)
            self.assertTrue(refs['complete'])
            return answer('The cached bibliography contains six entries; reading references alone does not verify their theorems.')

        runner, client, worker = self.runner(literature,
            [native('open_url', {'url': URL}), references, synthesis])
        with (patch.object(literature, '_fetch', return_value=(bibliography(6), 'text/html', URL)) as fetch,
              patch.object(literature, '_request_once', side_effect=AssertionError('Real HTTP forbidden'))):
            result = runner.run('Open this URL and list its references: ' + URL)
        self.assertEqual(result['status'], 'complete')
        self.assertIn('six entries', result['answer'])
        self.assertEqual(len(client.requests), 3)
        worker.assert_not_called()
        self.assertEqual(runner.state['children'], [])
        fetch.assert_called_once()
        names = {tool['function']['name'] for tool in client.requests[0]['tools']}
        self.assertTrue({'open_url', 'read_page', 'search_page', 'read_references'} <= names)
        self.assertEqual(opened[0]['source_url'], URL)
        self.assertTrue(opened[0]['text_sha256'])
        self.assertTrue(opened[0]['retrieved_at'])
        self.assertIn('page_id', opened[0])
        self.assertNotIn('[TRUNCATED', json.dumps(runner.state['messages']))

    def test_uncached_offline_url_returns_explicit_error_without_http(self):
        literature = LiteratureTools(self.root, online=False)

        def explain(payload):
            error = source_result(payload)['error']
            self.assertIsInstance(error, dict)
            self.assertIn('offline', error['message'].lower())
            self.assertFalse(error['retryable'])
            return answer('This URL is uncached and unavailable in offline mode.')

        runner, client, worker = self.runner(literature, [native('open_url', {'url': URL}), explain])
        with (patch.object(literature, '_fetch', side_effect=AssertionError('Offline fetch forbidden')) as fetch,
              patch.object(literature, '_request_once', side_effect=AssertionError('Real HTTP forbidden'))):
            result = runner.run('Read this uncached URL while offline: ' + URL)
        fetch.assert_not_called()
        worker.assert_not_called()
        self.assertEqual(literature.stats['requests'], 0)
        self.assertEqual(result['status'], 'complete')
        self.assertIn('offline mode', result['answer'])
        self.assertEqual(len(client.requests), 2)

    def test_workspace_reference_pagination_returns_every_entry_without_deduplicating_shared_urls(self):
        literature = LiteratureTools(self.root, online=True, max_chars=120000)
        workspace = Workspace(self.root, literature=literature)
        with (patch.object(literature, '_fetch', return_value=(bibliography(37), 'text/html', URL)),
              patch.object(literature, '_request_once', side_effect=AssertionError('Real HTTP forbidden'))):
            opened = json.loads(workspace.execute('open_url', {'url': URL}))
        entries, pages, start = [], [], 1
        while start is not None:
            self.assertLess(len(pages), 10, 'reference paging did not advance')
            raw = workspace.execute('read_references', {'page_id': opened['page_id'], 'start': start, 'limit': 8})
            self.assertNotIn('[TRUNCATED', raw)
            page = json.loads(raw)
            self.assertEqual(page['total'], 37)
            self.assertEqual(page['start'], start)
            self.assertEqual(page['end'] - page['start'] + 1, len(page['references']))
            entries.extend(page['references'])
            pages.append(page)
            if page['next_start'] is not None:
                self.assertGreater(page['next_start'], start)
                self.assertFalse(page['complete'])
            start = page['next_start']
        self.assertEqual(opened['reference_count'], 37)
        self.assertEqual(len(entries), 37)
        self.assertTrue(pages[-1]['complete'])
        self.assertTrue(pages[-1]['coverage_complete'])
        for number, entry in enumerate(entries, 1):
            self.assertIn(f'Exact reference title {number}.', json.dumps(entry))
        # The 12 references without href and the 25 references sharing one href
        # are bibliography entries in their own right, not a list of unique URLs.
        self.assertEqual(len(pages), 5)


if __name__ == '__main__':
    unittest.main()
