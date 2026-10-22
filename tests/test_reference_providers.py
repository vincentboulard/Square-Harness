"""Reference retrieval must preserve a usable work's identity across fallbacks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from mathagent.literature import LiteratureTools


CROSSREF = {'message': {'items': [
    {'DOI': '10.1214/aop/1176990437',
     'title': ['Some Potential Theory for Reflecting Brownian Motion in Holder and Lipschitz Domains'],
     'author': [{'given': 'Richard F.', 'family': 'Bass'}, {'given': 'Pei', 'family': 'Hsu'}],
     'issued': {'date-parts': [[1991, 4, 1]]}, 'type': 'journal-article',
     'container-title': ['The Annals of Probability'],
     'abstract': '<jats:p>Construction in Lipschitz domains.</jats:p>'}
]}}
OPENLIBRARY = {'docs': [
    {'key': '/works/OL123W', 'title': 'Stochastic Differential Equations',
     'author_name': ['Bernt Øksendal'], 'first_publish_year': 1985,
     'publisher': ['Springer', 'Springer']},
    {'key': '/works/OL124W', 'title': 'Probability: Theory and Examples',
     'author_name': ['Richard Durrett'], 'first_publish_year': 1991,
     'publisher': ['Wadsworth', 'Cambridge University Press']}
]}


class ReferenceProviderTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.lit = LiteratureTools(self.root, online=True)
        sleeper = patch('mathagent.literature.time.sleep')
        sleeper.start()
        self.addCleanup(sleeper.stop)

    def search(self, provider, query='reflecting Brownian motion Lipschitz domains', **kwargs):
        return json.loads(self.lit.execute('search_papers', {'provider': provider, 'query': query, **kwargs}))

    def reply(self, payload):
        return 200, {'content-type': 'application/json'}, json.dumps(payload, ensure_ascii=False).encode()

    def test_crossref_article_normalized_and_cached_without_extra_http_requests(self):
        with patch.object(self.lit, '_request_once', return_value=self.reply(CROSSREF)) as request:
            result = self.search('crossref', limit=4)
            record = result['results'][0]
            self.assertEqual(record['authors'], ['Richard F. Bass', 'Pei Hsu'])
            self.assertEqual((record['year'], record['type']), (1991, 'journal-article'))
            self.assertEqual(record['identifier'], 'doi:10.1214/aop/1176990437')
            self.assertEqual(record['url'], 'https://doi.org/10.1214/aop/1176990437')
            self.assertEqual(record['abstract'], 'Construction in Lipschitz domains.')
            self.assertEqual(result['provider'], 'crossref')
            params = parse_qs(urlsplit(request.call_args.args[0]).query)
            self.assertEqual(params['query.bibliographic'], ['reflecting Brownian motion Lipschitz domains'])
            self.assertEqual(params['rows'], ['4'])
            self.lit.max_requests = 1
            self.assertTrue(self.search('crossref', limit=4)['cached'])
            self.assertIn('budget', self.search('crossref', query='another query')['error'])
            self.assertEqual(request.call_count, 1)
            self.assertEqual(self.lit.stats['requests'], 1)

    def test_openlibrary_book_records_do_not_conflate_edition_publishers(self):
        with patch.object(self.lit, '_request_once', return_value=self.reply(OPENLIBRARY)) as request:
            result = self.search('openlibrary', query='Øksendal Stochastic Differential Equations', limit=2)
            a, b = result['results']
            self.assertEqual(a['authors'], ['Bernt Øksendal'])
            self.assertEqual((a['identifier'], a['title'], a['year'], a['type']),
                             ('openlibrary:OL123W', 'Stochastic Differential Equations', 1985, 'book'))
            self.assertEqual(a['year_kind'], 'catalog_first_publication')
            self.assertEqual(a['publisher'], 'Springer')
            self.assertIsNone(b['publisher'])
            self.assertIn('edition unspecified', b['source'])
            self.assertEqual(a['url'], 'https://openlibrary.org/works/OL123W')
            url = urlsplit(request.call_args.args[0])
            self.assertEqual((url.scheme, url.netloc, url.path), ('https', 'openlibrary.org', '/search.json'))
            self.assertEqual(parse_qs(url.query)['q'], ['Øksendal Stochastic Differential Equations'])

    def test_book_catalog_does_not_accept_an_arbitrary_record_link(self):
        payload = {'docs': [
            {'key': 'https://untrusted.example/book', 'title': 'Fake', 'author_name': ['Someone']},
            {'key': '/works/../admin', 'title': 'Fake'},
            {'key': '/works/OL126W', 'title': 'Valid', 'author_name': 'malformed string',
             'first_publish_year': True},
            {'key': '/works/OL127W', 'title': ''}, None]}
        with patch.object(self.lit, '_request_once', return_value=self.reply(payload)):
            records = self.search('openlibrary', limit=8)['results']
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['authors'], [])
        self.assertIsNone(records[0]['year'])

    def test_each_new_provider_obeys_offline_cache_and_refuses_uncached_network(self):
        for provider, payload in [('crossref', CROSSREF), ('openlibrary', OPENLIBRARY)]:
            with self.subTest(provider=provider):
                with patch.object(self.lit, '_request_once', return_value=self.reply(payload)):
                    first = self.search(provider)
                offline = LiteratureTools(self.root)
                with patch.object(offline, '_request_once') as request:
                    cached = json.loads(offline.execute('search_papers', {'provider': provider,
                                        'query': 'reflecting Brownian motion Lipschitz domains'}))
                    missing = json.loads(offline.execute('search_papers', {'provider': provider, 'query': 'uncached'}))
                    self.assertEqual(cached['results'], first['results'])
                    self.assertTrue(cached['cached'])
                    self.assertIn('Offline', missing['error'])
                    request.assert_not_called()

    def test_provider_failure_is_an_error_not_an_empty_success_or_cached_negative(self):
        for provider in ('crossref', 'openlibrary'):
            with self.subTest(provider=provider):
                with patch.object(self.lit, '_request_once', return_value=(502, {}, b'bad gateway')) as request:
                    failed = self.search(provider, query='failure ' + provider)
                    self.assertIn('HTTP 502', failed['error'])
                    self.assertNotIn('results', failed)
                    self.assertEqual(request.call_count, 1)
                with patch.object(self.lit, '_request_once', return_value=self.reply(
                        CROSSREF if provider == 'crossref' else OPENLIBRARY)) as request:
                    self.assertTrue(self.search(provider, query='failure ' + provider)['results'])
                    self.assertEqual(request.call_count, 1)

    def test_new_providers_are_advertised_and_validate_budget_before_network(self):
        provider_schema = self.lit.schemas()[0]['function']['parameters']['properties']['provider']['enum']
        self.assertIn('crossref', provider_schema)
        self.assertIn('openlibrary', provider_schema)
        self.lit.max_requests = 0
        with patch.object(self.lit, '_request_once') as request:
            for provider in ('crossref', 'openlibrary'):
                self.assertIn('budget', self.search(provider)['error'])
            request.assert_not_called()

    def test_description_truncation_preserves_complete_record_and_search_identity(self):
        result = {'provider': 'crossref', 'query': 'Bass Hsu reflecting Brownian motion',
                  'evidence_kind': 'discovery_metadata_not_verified_full_text',
                  'results': [{'identifier': 'doi:10.1214/aop/1176990437',
                               'title': CROSSREF['message']['items'][0]['title'][0],
                               'authors': ['Richard F. Bass', 'Pei Hsu'],
                               'year': 1991, 'type': 'journal-article',
                               'url': 'https://doi.org/10.1214/aop/1176990437',
                               'abstract': 'Descriptive source text. ' * 1000}]}
        before = json.loads(json.dumps(result))
        self.lit.max_chars = 750
        response = json.loads(self.lit._emit(result))
        self.assertTrue(response['truncated'])
        self.assertEqual(response['provider'], result['provider'])
        self.assertEqual(response['query'], result['query'])
        self.assertEqual(response['evidence_kind'], result['evidence_kind'])
        for key in ('identifier', 'title', 'authors', 'year', 'type', 'url'):
            self.assertEqual(response['results'][0][key], result['results'][0][key])
        self.assertLess(len(response['results'][0]['abstract']), len(result['results'][0]['abstract']))
        self.assertTrue(response['results'][0]['abstract_truncated'])
        self.assertEqual(result, before, 'emitting bounded evidence must not mutate cached source data')
        self.assertLessEqual(self.lit.stats['returned_chars'], 750)

    def test_low_budget_drops_complete_records_with_count_without_mutating_identity(self):
        records = [{'identifier': 'doi:10.1234/' + str(i), 'title': 'Distinctive complete title ' + str(i),
                    'authors': ['An Author'], 'url': 'https://doi.org/10.1234/' + str(i)} for i in range(5)]
        result = {'provider': 'crossref', 'query': 'a complete topic query', 'results': records}
        self.lit.max_chars = 400
        response = json.loads(self.lit._emit(result))
        self.assertEqual(response['provider'], 'crossref')
        self.assertEqual(response['query'], 'a complete topic query')
        self.assertGreater(response['results_omitted'], 0)
        self.assertEqual(len(response['results']) + response['results_omitted'], 5)
        self.assertEqual(response['results'], records[:len(response['results'])])
        self.assertEqual(len(records), 5)
        self.assertLessEqual(self.lit.stats['returned_chars'], 400)

    def test_too_small_identity_envelope_is_explicit_error_instead_of_fabricated_fragment(self):
        self.lit.max_chars = 150
        result = {'provider': 'openlibrary', 'query': 'a long topic ' * 30,
                  'results': [{'identifier': 'openlibrary:OL123W', 'title': 'A book'}]}
        response = json.loads(self.lit._emit(result))
        self.assertIn('budget exhausted', response['error'])
        self.assertNotIn('provider', response)
        self.assertNotIn('query', response)
        self.assertTrue(response['truncated'])
        self.assertLessEqual(self.lit.stats['returned_chars'], 150)


if __name__ == '__main__':
    unittest.main()
