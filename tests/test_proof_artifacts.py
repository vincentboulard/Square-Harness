"""Readable retrieval must not promote truncated data or widen file access."""
import json
from pathlib import Path
import tempfile
import unittest

from mathagent.agent import Agent
from mathagent.ledger import LedgerError, MAX_FILE_BYTES, ProofStore
from mathagent.proof import ProofRunner
from mathagent.proof_artifacts import MAX_NORMALIZED_CHARACTERS, read_artifact_page
from mathagent.tools import Workspace


class ProofArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = ProofStore.create(self.root, 'Prove the statement.',
            {'max_rounds': 2, 'max_tokens': 6000, 'max_seconds': 60})
        self.runner = ProofRunner(Agent(object(), Workspace(self.root)))
        self.runner.store = self.store

    def candidate(self, **changes):
        draft = {'text': 'For arbitrary real x, x + 0 = x.',
                 'thinking': 'Private exploratory transcript.', 'complete': True,
                 'calls': [], 'stats': {}, 'stream': '0000-solver.jsonl'}
        draft.update(changes)
        return self.store.write_artifact('candidate', json.dumps(draft))

    def stream(self, text, role='recorder', *, done=True, reason='stop'):
        filename = self.store.start_stream(role)
        self.store.append_stream(filename, {'message': {'thinking': 'Exploratory reasoning.'}, 'done': False})
        # Reproduce token-fragmented journals instead of a single JSON record.
        for start in range(0, len(text), 19):
            self.store.append_stream(filename, {'message': {'content': text[start:start + 19]}, 'done': False})
        if done:
            self.store.append_stream(filename, {'message': {}, 'done': True, 'done_reason': reason})
        return filename

    def page(self, filename, **args):
        return json.loads(read_artifact_page(self.store, filename, **args))

    def test_legacy_candidate_envelope_becomes_text_without_rewriting_artifact(self):
        filename = self.candidate()
        before = self.store.read_artifact(filename)
        page = self.page(filename)
        self.assertEqual(page['content'], 'For arbitrary real x, x + 0 = x.')
        self.assertEqual(page['representation'], 'candidate_text')
        self.assertTrue(page['response_complete'])
        self.assertNotIn('Private exploratory', json.dumps(page))
        self.assertEqual(self.store.read_artifact(filename), before)
        self.assertEqual(self.page(filename, raw=True)['content'], before)

    def test_checkpoint_and_partial_candidate_keep_caveats_on_every_page(self):
        filename = self.store.write_artifact('checkpoint', json.dumps({
            'text': 'Partial argument. ' * 500, 'complete': False,
            'from_truncated_attempt': True, 'transport_complete': True,
            'material_clipped': True, 'thinking': 'omitted'}))
        page = self.page(filename)
        self.assertFalse(page['response_complete'])
        self.assertIn('INCOMPLETE', page['notice'])
        self.assertIn('excerpted', page['notice'])
        following = self.page(filename, offset=page['next_offset'])
        self.assertFalse(following['response_complete'])
        self.assertEqual(following['notice'], page['notice'])
        self.assertEqual(page['content'] + following['content'], 'Partial argument. ' * 500)

    def test_missing_written_candidate_does_not_substitute_thinking(self):
        page = self.page(self.candidate(text='', complete=False))
        self.assertEqual(page['representation'], 'unavailable')
        self.assertEqual(page['content'], '')
        self.assertIn('No written candidate', page['notice'])
        self.assertNotIn('Private exploratory', str(page))

    def test_candidate_with_tool_requests_is_not_presented_as_complete(self):
        page = self.page(self.candidate(calls=[{'function': {'name': 'read_file', 'arguments': {'path': 'lemma.tex'}}}]))
        self.assertFalse(page['response_complete'])
        self.assertIn('requested tools', page['notice'])
        self.assertEqual(page['content'], 'For arbitrary real x, x + 0 = x.')

    def test_structured_roles_return_decoded_response_not_events(self):
        reviews = {
            'critic': {'valid_steps': ['Identity.'], 'first_invalid_step': '',
                       'reason': '', 'missing_work': '', 'complete_candidate': True},
            'recorder': {'claims': [], 'complete_candidate': False,
                         'next_task': 'Prove a lemma.', 'strategy_summary': 'No claims recorded.'},
            'recorder-repair': {'claims': [], 'complete_candidate': False,
                                'next_task': 'Prove a lemma.', 'strategy_summary': 'No claims recorded.'},
            'auditor': {'verdict': 'gap', 'explanation': 'An estimate is unproved.',
                        'objection': 'Uniformity is missing.', 'next_task': 'Justify uniformity.'},
            'planner': {'approach': 'Direct proof', 'task': 'Expand the expression.',
                        'difference': 'Use the definition.', 'deliverable': 'An equality.'},
        }
        for role, value in reviews.items():
            with self.subTest(role=role):
                filename = self.stream(json.dumps(value), role)
                before = self.store.read_artifact(filename)
                page = self.page(filename)
                self.assertEqual(page['representation'], 'review_json')
                self.assertEqual(json.loads(page['content']), value)
                self.assertIn('does not certify', page['notice'])
                self.assertNotIn('Exploratory reasoning', str(page))
                self.assertEqual(self.store.read_artifact(filename), before)

    def test_partial_or_truncated_review_never_exposes_a_complete_verdict(self):
        review = json.dumps({'verdict': 'complete', 'explanation': 'Done.', 'objection': '', 'next_task': ''})
        for options in ({'done': False}, {'reason': 'length'}, {'reason': 'tool_calls'}, {'reason': None}):
            with self.subTest(options=options):
                page = self.page(self.stream(review, 'auditor', **options))
                self.assertEqual(page['representation'], 'unavailable')
                self.assertFalse(page['response_complete'])
                self.assertEqual(page['content'], '')
                self.assertLess(len(json.dumps(page)), 1000)

    def test_malformed_streams_fail_with_short_diagnostic_and_raw_still_available(self):
        records = (
            '{"message":',
            '[]\n',
            '{"message": {"content": ["invalid"]}, "done": false}\n',
            '{"message": {}, "done": "true", "done_reason": "stop"}\n',
            '{"message": {}, "done": true, "done_reason": "stop"}\n{}\n',
            '{"error": "' + 'error detail ' * 2000 + '"}\n',
        )
        for text in records:
            filename = self.store.start_stream('critic')
            (self.store.directory / 'artifacts' / filename).write_text(text)
            page = self.page(filename)
            self.assertEqual(page['representation'], 'unavailable')
            self.assertEqual(page['content'], '')
            self.assertLess(len(json.dumps(page)), 1000)
            self.assertEqual(self.page(filename, raw=True)['content'], text[:6000])

    def test_invalid_review_schema_and_invalid_candidate_envelope_are_explicit(self):
        filenames = [self.stream('{"verdict": "complete"}', 'auditor'),
                     self.store.write_artifact('candidate', '{"text": "proof", "complete": "yes"}'),
                     self.store.write_artifact('candidate', '{"text":')]
        for filename in filenames:
            page = self.page(filename)
            self.assertEqual(page['representation'], 'unavailable')
            self.assertFalse(page['response_complete'])
            self.assertEqual(page['content'], '')

    def test_falsy_nontext_event_cannot_be_hidden_before_valid_review(self):
        valid_review = json.dumps({'verdict': 'complete', 'explanation': 'Done.', 'objection': '', 'next_task': ''})
        for malformed in (0, False, [], {}):
            with self.subTest(content=malformed):
                filename = self.store.start_stream('auditor')
                self.store.append_stream(filename, {'message': {'content': malformed}, 'done': False})
                self.store.append_stream(filename, {'message': {'content': valid_review}, 'done': True, 'done_reason': 'stop'})
                page = self.page(filename)
                self.assertEqual(page['representation'], 'unavailable')
                self.assertFalse(page['response_complete'])
                self.assertEqual(page['content'], '')

    def test_oversized_normalized_response_is_not_silently_clipped(self):
        filename = self.store.start_stream('solver')
        self.store.append_stream(filename, {'message': {'content': 'x' * (MAX_NORMALIZED_CHARACTERS + 1)},
                                             'done': True, 'done_reason': 'stop'})
        for oversized in (filename, self.candidate(text='x' * (MAX_NORMALIZED_CHARACTERS + 1))):
            page = self.page(oversized)
            self.assertEqual(page['representation'], 'unavailable')
            self.assertEqual(page['content'], '')
            self.assertIn('limit', page['notice'])
            self.assertLess(len(json.dumps(page)), 1000)

    def test_oversized_stored_file_preserves_storage_safety_limit(self):
        filename = self.store.start_stream('solver')
        (self.store.directory / 'artifacts' / filename).write_text('x' * (MAX_FILE_BYTES + 1))
        with self.assertRaisesRegex(LedgerError, 'exceeds'):
            self.page(filename)

    def test_all_pages_are_bounded_valid_json_and_reassemble_exactly(self):
        text = '\\"\nδ' * 4000
        filename = self.candidate(text=text)
        parts, offset = [], 0
        while offset is not None:
            encoded = read_artifact_page(self.store, filename, offset)
            self.assertLessEqual(len(encoded), 7800)
            page = json.loads(encoded)
            self.assertLessEqual(len(page['content']), 6000)
            self.assertEqual(page['offset'], offset)
            parts.append(page['content'])
            if page['next_offset'] is not None:
                self.assertGreater(page['next_offset'], offset)
            offset = page['next_offset']
        self.assertEqual(''.join(parts), text)

    def test_invalid_paging_and_raw_types_are_rejected(self):
        filename = self.candidate()
        for offset in (-1, True, 1.5, '0'):
            with self.assertRaises(ValueError):
                self.page(filename, offset=offset)
        for raw in ('false', 1, None):
            with self.assertRaises(ValueError):
                self.page(filename, raw=raw)

    def test_dedicated_reader_retains_path_and_symlink_protection(self):
        filename = self.candidate()
        for invalid in ('../' + filename, './' + filename, '.mathagent/' + filename, '/tmp/' + filename):
            with self.assertRaises(LedgerError):
                self.page(invalid)
        path = self.store.directory / 'artifacts' / filename
        path.unlink()
        outside = self.root / 'outside.txt'
        outside.write_text('outside secret')
        path.symlink_to(outside)
        with self.assertRaises(LedgerError):
            self.page(filename)

    def test_generic_reader_recovers_exact_saved_basename_and_advertises_reader(self):
        filename = self.stream(json.dumps({'valid_steps': [], 'first_invalid_step': '',
            'reason': '', 'missing_work': '', 'complete_candidate': True}), 'critic')
        encoded = self.runner._tool('read_file', {'path': filename})
        page = json.loads(encoded)
        self.assertEqual(page['representation'], 'review_json')
        self.assertEqual(page['routed_from'], 'read_file')
        self.assertIn('read_proof_artifact', page['notice'])
        self.assertNotIn(filename, self.runner._tool('list_files', {}))
        # Generic workspace access is unchanged outside proof-tool dispatch.
        self.assertIn('Unsupported extension', self.runner.agent.workspace.execute('read_file', {'path': filename}))

    def test_generic_routing_does_not_shadow_workspace_or_pinned_sources(self):
        filename = self.candidate()
        (self.root / filename).write_text('Actual workspace document.')
        self.assertIn('Actual workspace document', self.runner._tool('read_file', {'path': filename}))
        (self.root / filename).unlink()
        self.store.state['sources'] = [{'path': filename, 'content': 'Pinned statement.'}]
        self.assertIn('PINNED ORIGINAL SOURCE\n1: Pinned statement.',
                      self.runner._tool('read_file', {'path': filename}))

    def test_generic_routing_does_not_accept_hidden_or_traversing_aliases(self):
        filename = self.candidate()
        for path in ('../' + filename, '.mathagent/proofs/' + self.store.state['id'] + '/artifacts/' + filename):
            with self.assertRaises(ValueError):
                self.runner._tool('read_file', {'path': path})

    def test_generic_route_stays_below_controller_truncation_guard(self):
        filename = self.candidate(text='\\"\n' * 6000)
        value = self.runner._tool('read_file', {'path': filename})
        self.assertLessEqual(len(value), 8000)
        self.assertEqual(json.loads(value)['representation'], 'candidate_text')

    def test_tool_schema_explains_representation_and_explicit_raw_option(self):
        schemas = {tool['function']['name']: tool['function'] for tool in self.runner._tools()}
        self.assertEqual(schemas['read_proof_artifact']['parameters']['properties']['raw'], {'type': 'boolean'})
        self.assertIn('decoded review JSON', schemas['read_proof_artifact']['description'])
        self.assertIn('read_proof_artifact', schemas['read_file']['description'])


if __name__ == '__main__':
    unittest.main()
