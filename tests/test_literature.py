import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from mathagent.literature import LiteratureError, LiteratureTools, _HTMLText

ATOM = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2401.12345v2</id><title>A useful theorem</title>
<summary>A result under explicit assumptions.</summary><author><name>A. Author</name></author>
<published>2024-01-01</published><updated>2024-02-01</updated></entry></feed>'''
HTML = b'''<!doctype html><html><head><title>Some paper</title><script>ignore rules</script></head>
<body><article class="ltx_document"><h1>1 Introduction</h1><p>First line.</p>
<h2>Lemma 1</h2><p>For every x, <math alttext="x^2 \\geq 0"><mi>broken duplicate</mi></math>.</p>
<h2>References</h2><p>A reference.</p></article></body></html>'''


class LiteratureTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.lit = LiteratureTools(self.root, online=True)
        self.sleep = patch('mathagent.literature.time.sleep')
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def result(self, name, args):
        return json.loads(self.lit.execute(name, args))

    def open_html(self):
        with patch.object(self.lit, '_request_once', return_value=(200, {'content-type': 'text/html'}, HTML)):
            return self.result('open_paper', {'identifier_or_url': '2401.12345v2'})

    def test_default_offline_all_direct_network_paths_blocked(self):
        lit = LiteratureTools(self.root)
        with patch.object(lit, '_request_once') as request, patch('socket.getaddrinfo') as dns:
            for name, args in [('search_papers', {'query': 'test'}), ('search_web', {'query': 'test'}),
                               ('open_paper', {'identifier_or_url': '2401.12345'})]:
                self.assertIn('Offline', json.loads(lit.execute(name, args))['error'])
            with self.assertRaisesRegex(LiteratureError, 'Offline'):
                lit._fetch('https://example.org/paper')
            request.assert_not_called()
            dns.assert_not_called()
        self.assertEqual(lit.stats['requests'], 0)

    def test_arxiv_search_metadata_cached_offline(self):
        with patch.object(self.lit, '_request_once', return_value=(200, {'content-type': 'application/atom+xml'}, ATOM)) as request:
            first = self.result('search_papers', {'query': 'observability'})
            self.assertEqual(first['results'][0]['identifier'], 'arxiv:2401.12345v2')
            self.assertEqual(first['evidence_kind'], 'discovery_metadata_not_verified_full_text')
            second = self.result('search_papers', {'query': 'observability'})
            self.assertTrue(second['cached'])
            self.assertEqual(request.call_count, 1)
        offline = LiteratureTools(self.root)
        with patch.object(offline, '_request_once') as request:
            self.assertTrue(json.loads(offline.execute('search_papers', {'query': 'observability'}))['cached'])
            request.assert_not_called()

    def test_provider_misspelling_not_silently_fallback(self):
        with patch.object(self.lit, '_request_once') as request:
            self.assertIn('Unknown provider', self.result('search_papers', {'query': 'x', 'provider': 'arxvi'})['error'])
            request.assert_not_called()

    def test_schema_names_and_disabled_code_path(self):
        self.assertEqual({x['function']['name'] for x in self.lit.schemas()}, {'search_papers', 'search_web', 'open_paper', 'read_paper', 'search_paper'})
        self.assertIn('Unknown', self.result('__class__', {})['error'])
        self.assertIn('error', self.result('search_papers', 'not json'))
        self.assertIn('error', self.result('search_papers', {'query': 'x', 'extra': 1}))

    def test_provider_credentials_only_headers_and_not_snapshots(self):
        payload = {'data': [{'paperId': 'abc', 'title': 'Test', 'authors': [{'name': 'A'}], 'externalIds': {'ArXiv': '2401.12345'}, 'openAccessPdf': {'url': 'https://example.org/a.pdf'}}]}
        with patch.dict('os.environ', {'SEMANTIC_SCHOLAR_API_KEY': 'sensitive-test-value'}), patch.object(self.lit, '_request_once', return_value=(200, {}, json.dumps(payload).encode())) as request:
            result = self.result('search_papers', {'query': 'lemma', 'provider': 'semantic_scholar'})
            self.assertEqual(result['results'][0]['open_access_url'], 'https://example.org/a.pdf')
            url, headers, _ = request.call_args.args
            self.assertNotIn('sensitive', url)
            self.assertEqual(headers['x-api-key'], 'sensitive-test-value')
        self.assertNotIn('sensitive-test-value', json.dumps(self.lit.snapshot()))
        for p in (self.root / '.mathagent/literature').rglob('*.json'):
            self.assertNotIn('sensitive-test-value', p.read_text())

    def test_openalex_abstract_and_auth(self):
        payload = {'results': [{'id': 'https://openalex.org/W1', 'title': 'A theorem', 'abstract_inverted_index': {'world': [1], 'Hello': [0]}, 'authorships': [], 'best_oa_location': {'pdf_url': 'https://example.org/p.pdf'}}]}
        with patch.dict('os.environ', {'OPENALEX_API_KEY': 'secret-key'}), patch.object(self.lit, '_request_once', return_value=(200, {}, json.dumps(payload).encode())) as request:
            result = self.result('search_papers', {'query': 'theorem', 'provider': 'openalex'})
            self.assertEqual(result['results'][0]['abstract'], 'Hello world')
            self.assertEqual(request.call_args.args[1]['Authorization'], 'Bearer secret-key')

    def test_web_requires_key_but_cached_works_without_key(self):
        with patch.dict('os.environ', {}, clear=True), patch.object(self.lit, '_request_once') as request:
            self.assertIn('BRAVE_SEARCH_API_KEY', self.result('search_web', {'query': 'x'})['error'])
            request.assert_not_called()
        payload = b'{"web":{"results":[{"title":"Test","url":"https://example.org","description":"Good"}]}}'
        with patch.dict('os.environ', {'BRAVE_SEARCH_API_KEY': 'secret'}), patch.object(self.lit, '_request_once', return_value=(200, {}, payload)):
            self.assertEqual(self.result('search_web', {'query': 'x'})['results'][0]['title'], 'Test')
        with patch.dict('os.environ', {}, clear=True):
            self.assertTrue(self.result('search_web', {'query': 'x'})['cached'])

    def test_paper_cache_math_exact_citations_and_offline_reads(self):
        opened = self.open_html()
        document_id = opened['document_id']
        self.assertEqual(opened['version'], 'v2')
        self.assertEqual(opened['sha256'], hashlib.sha256(HTML).hexdigest())
        passage = self.result('read_paper', {'document_id': document_id})
        self.assertIn('$x^2 \\geq 0$', passage['passage'])
        self.assertNotIn('broken duplicate', passage['passage'])
        self.assertNotIn('ignore rules', passage['passage'])
        self.assertTrue(passage['citation'].startswith(document_id + ':L'))
        self.assertTrue(opened['sections'])
        self.lit.online = False
        cached = self.result('open_paper', {'identifier_or_url': 'https://arxiv.org/pdf/2401.12345v2.pdf'})
        self.assertEqual(cached['document_id'], document_id)
        self.assertTrue(cached['cached'])
        matches = self.result('search_paper', {'document_id': document_id, 'query': 'every'})
        self.assertEqual(matches['matches'][0]['citation'], document_id + ':L5')

    def test_mathml_without_alttext_is_not_discarded(self):
        parser = _HTMLText()
        parser.feed('<p>Equation <math><mi>x</mi><mo>+</mo><mn>1</mn></math></p>')
        self.assertIn('x+1', parser.text())

    def test_arxiv_missing_html_uses_pdf(self):
        replies = [(404, {}, b''), (200, {'content-type': 'application/pdf'}, b'%PDF sample')]
        with patch.object(self.lit, '_request_once', side_effect=replies) as request, patch.object(self.lit, '_pdf', return_value=('Lemma\nProof', 'test extractor')):
            result = self.result('open_paper', {'identifier_or_url': '2401.12345v1'})
        self.assertEqual(result['source_url'], 'https://arxiv.org/pdf/2401.12345v1')
        self.assertEqual(request.call_count, 2)

    def test_arxiv_404_fallback_obeys_budget(self):
        self.lit.reset_budget(max_requests=1)
        with patch.object(self.lit, '_request_once', return_value=(404, {}, b'')) as request:
            self.assertIn('budget', self.result('open_paper', {'identifier_or_url': '2401.12345'})['error'])
            self.assertEqual(request.call_count, 1)

    def test_urls_blocked_before_network(self):
        urls = ['http://example.org/p', 'file:///etc/passwd', 'https://user:pass@example.org/p',
                'https://127.0.0.1/p', 'https://[::1]/p', 'https://169.254.169.254/latest',
                'https://10.1.2.3/p', 'https://localhost/p', 'https://host.internal/p',
                'https://example.org:8443/p', 'https://example.org\\@127.0.0.1/p',
                'https://example.org/\nheader', 'https://[fe80::1%25eth0]/p']
        with patch.object(self.lit, '_request_once') as request:
            for url in urls:
                with self.subTest(url=url):
                    self.assertIn('error', self.result('open_paper', {'identifier_or_url': url}))
            request.assert_not_called()

    def test_dns_private_and_mixed_addresses_blocked(self):
        for addresses in (['127.0.0.1'], ['93.184.216.34', '10.0.0.2']):
            with patch('socket.getaddrinfo', return_value=[(2, 1, 6, '', (a, 443)) for a in addresses]), patch('socket.create_connection') as connect:
                with self.assertRaisesRegex(LiteratureError, 'nonpublic'):
                    self.lit._request_once('https://example.org/p', {}, 100)
                connect.assert_not_called()

    def test_https_connection_is_pinned_to_validated_ip(self):
        with patch.object(self.lit, '_addresses', return_value=['93.184.216.34']), patch('socket.create_connection') as connect, patch('ssl.create_default_context') as context, patch('http.client.HTTPSConnection') as connection:
            response = connection.return_value.getresponse.return_value
            response.status = 200
            response.getheaders.return_value = [('content-type', 'text/plain')]
            response.read.side_effect = [b'ok', b'']
            result = self.lit._request_once('https://example.org/p', {}, 100)
            connect.assert_called_once_with(('93.184.216.34', 443), timeout=15.0)
            context.return_value.wrap_socket.assert_called_once_with(connect.return_value, server_hostname='example.org')
            self.assertEqual(result[2], b'ok')

    def test_redirect_private_destination_blocked(self):
        with patch.object(self.lit, '_request_once', return_value=(302, {'location': 'https://127.0.0.1/private'}, b'')) as request:
            self.assertIn('error', self.result('open_paper', {'identifier_or_url': 'https://example.org/p'}))
            self.assertEqual(request.call_count, 1)

    def test_redirect_never_leaks_provider_key(self):
        with patch.object(self.lit, '_request_once', side_effect=[(302, {'location': 'https://example.org/x'}, b''), (200, {}, b'ok')]) as request:
            self.lit._fetch('https://api.openalex.org/works', headers={'Authorization': 'Bearer secret'})
            self.assertEqual(request.call_args_list[1].args[1], {})

    def test_redirects_are_bounded_and_counted(self):
        with patch.object(self.lit, '_request_once', return_value=(302, {'location': '/again'}, b'')) as request:
            self.assertIn('redirect', self.result('open_paper', {'identifier_or_url': 'https://example.org/p'})['error'])
            self.assertEqual(request.call_count, 4)
            self.assertEqual(self.lit.stats['requests'], 4)

    def test_retry_429_bounded_and_counted(self):
        with patch.object(self.lit, '_request_once', return_value=(429, {'retry-after': '1'}, b'')) as request:
            self.assertIn('HTTP 429', self.result('search_papers', {'query': 'x'})['error'])
            self.assertEqual(request.call_count, 2)
            self.assertEqual(self.lit.stats['requests'], 2)

    def test_long_backoff_does_not_sleep(self):
        with patch.object(self.lit, '_request_once', return_value=(429, {'retry-after': '600'}, b'')), patch('time.sleep') as sleep:
            self.assertIn('rate limited', self.result('search_papers', {'query': 'x'})['error'])
            sleep.assert_not_called()

    def test_rate_limit_between_uncached_arxiv_requests(self):
        with patch.object(self.lit, '_request_once', return_value=(200, {}, ATOM)), patch('time.sleep') as sleep:
            self.result('search_papers', {'query': 'x'})
            self.result('search_papers', {'query': 'y'})
            self.assertGreater(sleep.call_args.args[0], 2.0)

    def test_budget_request_reserved_and_persisted_before_dispatch(self):
        events = []
        self.lit.on_budget_change = lambda: events.append(('save', self.lit.stats['requests']))
        def request(*args):
            events.append(('request', self.lit.stats['requests']))
            return 200, {}, ATOM
        self.lit.reset_budget(max_requests=1)
        with patch.object(self.lit, '_request_once', side_effect=request):
            self.result('search_papers', {'query': 'x'})
            self.assertIn('budget', self.result('search_papers', {'query': 'y'})['error'])
        self.assertEqual(events[:2], [('save', 1), ('request', 1)])
        self.assertEqual(self.lit.stats['requests'], 1)

    def test_resume_keeps_spent_budget_and_never_enables_online(self):
        self.lit.stats = {'requests': 7, 'returned_chars': 1300}
        snapshot = self.lit.snapshot()
        offline = LiteratureTools(self.root, max_requests=20, max_chars=40000)
        offline.restore(snapshot)
        self.assertFalse(offline.online)
        self.assertEqual(offline.stats, self.lit.stats)
        self.assertEqual(offline.max_requests, 12)
        offline.restore({'online': True, 'stats': {'requests': 2, 'returned_chars': 2}})
        self.assertEqual(offline.stats['requests'], 7)
        self.lit.restore({'online': False})
        self.assertFalse(self.lit.online)

    def test_small_budget_outputs_valid_json(self):
        self.lit.reset_budget(max_chars=250)
        with patch.object(self.lit, '_request_once', return_value=(200, {}, ATOM)):
            result = self.lit.execute('search_papers', {'query': 'x'})
            self.assertIsInstance(json.loads(result), dict)
            self.assertLessEqual(len(result), 250)
        self.assertLessEqual(self.lit.stats['returned_chars'], 250)

    def test_budget_exhaustion_stays_json_without_more_network(self):
        self.lit.reset_budget(max_chars=1)
        with patch.object(self.lit, '_request_once', return_value=(200, {}, ATOM)) as request:
            result = self.result('search_papers', {'query': 'x'})
            self.assertIn('budget exhausted', result['error'])
            # No reason to retrieve evidence when not even an envelope fits.
            self.assertLessEqual(request.call_count, 1)

    def test_global_deadline_blocks_dispatch_and_backoff(self):
        self.lit.deadline = time.monotonic() - 1
        with patch.object(self.lit, '_request_once') as request:
            self.assertIn('time budget', self.result('search_papers', {'query': 'x'})['error'])
            request.assert_not_called()
        self.lit.deadline = time.monotonic() + 1
        with patch.object(self.lit, '_request_once', return_value=(429, {'retry-after': '3'}, b'')) as request, patch('time.sleep') as sleep:
            self.assertIn('time budget', self.result('search_papers', {'query': 'x'})['error'])
            self.assertEqual(request.call_count, 1)
            sleep.assert_not_called()

    def test_large_budget_truncation_marks_passage_unusable_as_full_range(self):
        (self.root / 'paper.txt').write_text('Long line mathematics ' * 1000)
        meta = self.lit.import_local('paper.txt')
        self.lit.reset_budget(max_chars=600)
        response = self.result('read_paper', {'document_id': meta['document_id']})
        self.assertTrue(response['truncated'])
        self.assertLessEqual(self.lit.stats['returned_chars'], 600)

    def test_remote_errors_do_not_expose_credentials(self):
        with patch.object(self.lit, '_request_once', side_effect=OSError('secret-token-private-proxy-url')):
            result = self.result('search_papers', {'query': 'x'})
        self.assertNotIn('secret', json.dumps(result))

    def test_pdf_without_reader_returns_actionable_error(self):
        with patch('shutil.which', return_value=None), patch('importlib.util.find_spec', return_value=None):
            with self.assertRaisesRegex(LiteratureError, 'poppler-utils'):
                self.lit._pdf(b'%PDF')

    def test_pdf_subprocess_timeout_is_safe(self):
        import subprocess
        with patch('shutil.which', return_value='/usr/bin/pdftotext'), patch('subprocess.run', side_effect=subprocess.TimeoutExpired('pdftotext', 1)):
            with self.assertRaisesRegex(LiteratureError, 'timed out'):
                self.lit._pdf(b'%PDF')

    def test_cache_symlinks_refused(self):
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        (self.root / '.mathagent').symlink_to(outside.name)
        with patch.object(self.lit, '_request_once') as request:
            self.assertIn('Symlinks', self.result('search_papers', {'query': 'x'})['error'])
            request.assert_not_called()
        self.assertEqual(list(Path(outside.name).iterdir()), [])

    def test_cache_file_symlink_and_tampering_refused(self):
        document = self.open_html()
        p = self.root / '.mathagent/literature/documents' / (document['document_id'] + '.txt')
        p.write_text('Changed proof')
        self.assertIn('hash', self.result('read_paper', {'document_id': document['document_id']})['error'])
        p.unlink()
        p.symlink_to('/etc/passwd')
        self.assertIn('Symlinks', self.result('read_paper', {'document_id': document['document_id']})['error'])
        self.assertIn('Invalid document_id', self.result('read_paper', {'document_id': '../../etc/passwd'})['error'])

    def test_import_local_is_offline_versioned_and_never_uploaded(self):
        self.lit.online = False
        manuscript = self.root / 'paper.tex'
        manuscript.write_text('Statement\nProof\n')
        with patch.object(self.lit, '_request_once') as request:
            first = self.lit.import_local('paper.tex')
            second = self.lit.import_local('paper.tex')
            self.assertEqual(first['document_id'], second['document_id'])
            self.assertTrue(second['cached'])
            manuscript.write_text('Statement changed\nProof\n')
            third = self.lit.import_local('paper.tex')
            self.assertNotEqual(first['document_id'], third['document_id'])
            self.assertEqual(first['source_url'], 'workspace:paper.tex')
            request.assert_not_called()
        self.assertIn('1: Statement', self.lit.read_paper(first['document_id'])['passage'])

    def test_local_import_symlink_parent_and_absolute_refused(self):
        (self.root / 'secret.tex').symlink_to('/etc/passwd')
        for name in ('../paper.tex', '/etc/passwd', '.private.tex', 'secret.tex'):
            with self.subTest(name=name), self.assertRaises(LiteratureError):
                self.lit.import_local(name)

    def test_local_pdf_uses_same_cache_reader(self):
        (self.root / 'paper.pdf').write_bytes(b'%PDF sample')
        self.lit.online = False
        with patch.object(self.lit, '_pdf', return_value=('Statement\nProof', 'mock pdf')):
            meta = self.lit.import_local('paper.pdf')
        self.assertEqual(meta['extractor'], 'mock pdf')
        self.assertIn('2: Proof', self.lit.read_paper(meta['document_id'])['passage'])

    def test_oversize_response_content_length_rejected(self):
        with patch.object(self.lit, '_addresses', return_value=['93.184.216.34']), patch('socket.create_connection'), patch('ssl.create_default_context'), patch('http.client.HTTPSConnection') as connection:
            response = connection.return_value.getresponse.return_value
            response.status = 200
            response.getheaders.return_value = [('content-length', '99999999')]
            with self.assertRaisesRegex(LiteratureError, 'byte limit'):
                self.lit._request_once('https://example.org/p', {}, 100)
            response.read.assert_not_called()

    def test_long_html_wrapped_for_readable_line_retrieval(self):
        parser = _HTMLText()
        parser.feed('<p>' + 'mathematics ' * 2000 + '</p>')
        self.assertGreater(len(parser.text().splitlines()), 100)
        self.assertLess(max(map(len, parser.text().splitlines())), 181)


if __name__ == '__main__':
    unittest.main()
