"""The three review workflows (quick, journal, explain) with a fake model that answers by schema."""
import copy
import json
from pathlib import Path
import re
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from mathagent.agent import Agent
from mathagent.research import ResearchRunner
from mathagent.review import _prose_comment, EFFORT, ExplainRunner, JournalRunner, QuickReviewRunner, SummaryReviewRunner, level_for, runner_for_state
from mathagent.tools import Workspace

WRONG = """Lemma. Let (u_n) be a bounded sequence in H^1(0,1). Then a subsequence converges strongly in L^2(0,1).

Proof. Since H^1(0,1) is a Hilbert space, a subsequence converges weakly in H^1.
The embedding of H^1(0,1) into L^2(0,1) is continuous, hence the subsequence converges strongly in L^2.
This proves the lemma.
"""

PAPER = r"""\documentclass{amsart}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{definition}[theorem]{Definition}
\begin{document}
\begin{abstract}
We prove compactness of bounded sequences.
\end{abstract}
\section{Introduction}
Let $H^1(0,1)$ be the Sobolev space.
\begin{theorem}\label{thm:main}
Every bounded sequence in $H^1(0,1)$ has a subsequence converging in $L^2(0,1)$.
\end{theorem}
\section{Preliminaries}
\begin{definition}\label{def:h1}
$H^1(0,1)$ is the space of $L^2$ functions with an $L^2$ weak derivative.
\end{definition}
\begin{lemma}\label{lem:embed}
The embedding $H^1(0,1)\hookrightarrow L^2(0,1)$ is compact.
\end{lemma}
\begin{proof}
This is Rellich's theorem \cite[Theorem 9.16]{brezis}.
\end{proof}
\section{Proof of the main result}
\begin{proof}[Proof of Theorem~\ref{thm:main}]
By Definition~\ref{def:h1} and Lemma~\ref{lem:embed}, the sequence is bounded in a Hilbert space.
Since the embedding is continuous, the sequence converges strongly in $L^2$.
\end{proof}
\appendix
\section{Auxiliary}
\begin{lemma}\label{lem:aux}
An auxiliary fact.
\end{lemma}
\begin{proof}
Obvious.
\end{proof}
\begin{thebibliography}{9}
\bibitem{brezis} H. Brezis. Functional Analysis, Sobolev Spaces and PDE. Springer, 2011.
\end{thebibliography}
\end{document}
"""


def line_of(prompt, phrase):
    match = re.search(r'L(\d+): [^\n]*' + re.escape(phrase), prompt)
    return int(match.group(1)) if match else 0


def default_check(prompt, schema):
    """The fake verifier flags the 'continuous, hence … strongly' step wherever it appears."""
    n = line_of(prompt, 'continuous, hence') or line_of(prompt, 'embedding is continuous')
    proof = re.search(r'\nPROOF[^\n]*:\n(.*?)(?:\n\n|\Z)', prompt, re.S)
    issues = []
    if n and proof and f'L{n}:' in proof.group(1):
        issue = {'start_line': n, 'end_line': n, 'quote': 'continuous, hence' if 'continuous, hence' in prompt else 'embedding is continuous',
                 'kind': 'invalid_inference', 'evidence': 'Continuity of the embedding does not turn weak into strong convergence; compactness (Rellich) is needed.',
                 'suggestion': 'Use the compactness of the embedding (Rellich–Kondrachov).'}
        if 'severity' in json.dumps(schema):
            issue['severity'] = 'major'
        issues.append(issue)
    value = {'explanation': 'Checked each step.', 'issues': issues, 'verdict': 'issues_found' if issues else 'no_issue_found'}
    if 'external' in schema['properties']:
        value['external'] = [{'key': 'brezis', 'place': 'Theorem 9.16', 'used_for': 'Rellich compactness theorem in one dimension'}] if 'cite' in prompt or 'Rellich' in prompt else []
    return value


def default_reply(payload, overrides):
    prompt = payload['messages'][1]['content']
    schema = payload.get('format') or {}
    props = set(schema.get('properties', {}))
    for name, test in (('confirm', {'assessment', 'argument'}), ('check', {'explanation', 'issues', 'verdict'}),
                       ('explain', {'says', 'idea'}), ('fill', {'steps'}), ('explain_check', {'problems'}),
                       ('scope', {'contribution', 'main'}), ('write', {'summary', 'recommendation'}),
                       ('novelty', {'assessment', 'related'}), ('read', {'summary', 'comments'}),
                       ('extract', {'proof_start'}), ('pick', {'unit'})):
        if test <= props and (name != 'fill' or props == {'steps'}):
            if name in overrides:
                return overrides[name](prompt, schema)
            if name == 'check':
                return default_check(prompt, schema)
            if name == 'confirm':
                return {'assessment': 'valid', 'kind': 'invalid_inference', 'argument': 'Weak convergence plus a continuous map gives only weak convergence.',
                        'suggestion': 'Invoke Rellich.'}
            if name == 'scope':
                ids = re.findall(r'(U\d+): Theorem', prompt)
                return {'overview': 'The paper proves Rellich compactness on (0,1) from a compact embedding lemma.',
                        'field': 'Sobolev spaces', 'contribution': 'Compactness of bounded sequences.', 'main': ids[:1],
                        'notation': [{'start_line': 10, 'end_line': 10}], 'closest': ['brezis'], 'queries': ['Rellich compactness Sobolev']}
            if name == 'write':
                return {'summary': 'The paper proves a compactness result.', 'significance': 'Classical.', 'correctness': 'See Major 1.',
                        'presentation': 'Clear.', 'recommendation': 'major_revision', 'reasons': 'Major 1 invalidates the proof.',
                        'confidential': 'Only the main proof and its lemma were checked.'}
            if name == 'explain_check':
                return {'problems': []}
            if name == 'novelty':
                return {'assessment': 'The result is classical (result 1).', 'related': [{'n': 1, 'relation': 'textbook source'}]}
            if name == 'extract':
                return {'statement_start': 1, 'statement_end': 1, 'proof_start': 0, 'proof_end': 0}
            if name == 'read':
                n = line_of(prompt, 'Sobolev space')
                return {'summary': 'This part sets up $H^1(0,1)$ and states the main theorem.',
                        'comments': [{'start_line': n, 'end_line': n, 'quote': 'be the Sobolev space', 'kind': 'definition',
                                      'comment': 'Say which norm.', 'suggestion': 'Add the norm.'}] if n else []}
            raise AssertionError('No fake reply for ' + name)
    raise AssertionError('Unexpected request: ' + json.dumps(sorted(props)))


class FakeClient:
    host = 'http://fake.invalid'
    timeout = 60

    def __init__(self, overrides=None, fail_on=None, delay=0):
        self.overrides = overrides or {}
        self.requests = []
        self.fail_on = fail_on  # (role keyword in prompt, call number) → KeyboardInterrupt
        self.delay = delay
        self.lock = threading.Lock()

    def stream(self, payload):
        with self.lock:
            self.requests.append(copy.deepcopy(payload))
            count = len(self.requests)
        if self.fail_on and self.fail_on(payload, count):
            raise KeyboardInterrupt
        if self.delay:
            time.sleep(self.delay)
        value = default_reply(payload, self.overrides)
        text = value if isinstance(value, str) else json.dumps(value)
        yield {'message': {'content': text}, 'done': True, 'done_reason': 'stop',
               'eval_count': min(50, payload['options']['num_predict']), 'prompt_eval_count': 200}


class ReviewTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def agent(self, client, ctx=16384, predict=4096):
        workspace = Workspace(self.root)
        workspace.literature = None
        agent = Agent(client, workspace, ctx=ctx, predict=predict)
        agent.seed = 7
        return agent

    def kinds(self, client):
        return [sorted(r.get('format', {}).get('properties', {})) for r in client.requests]


class QuickTests(ReviewTestCase):
    def test_planted_error_is_found_confirmed_and_located(self):
        client = FakeClient()
        runner = QuickReviewRunner(self.agent(client))
        result = runner.start(WRONG, max_rounds=3)
        self.assertEqual(result['status'], 'reviewed')
        self.assertEqual(runner.state['verdict'], 'issues_found')
        self.assertEqual(runner.state['level'], 'medium')
        self.assertEqual([s['id'] for s in runner.state['sources']], ['P1'])
        self.assertEqual(runner.state['unit']['proofs'], [[3, 5]])
        self.assertEqual(len(runner.state['checks']), 2)  # medium: line by line + adversarial
        self.assertEqual(len(runner.state['confirms']), 1)
        report = result['report']
        self.assertIn('Issues found', report)
        self.assertIn('[P1:L4]', report)
        self.assertIn('confirmed by an independent re-check; raised by 2 of 2 passes', report)
        self.assertIn('Rellich', report)
        self.assertIn('L4: The embedding', report)  # the numbered proof in the appendix
        lenses = [r['messages'][1]['content'] for r in client.requests if 'LENS FOR THIS PASS' in r['messages'][1]['content']]
        self.assertTrue(any('LINE BY LINE' in p for p in lenses) and any('ADVERSARIAL' in p for p in lenses))

    def test_a_server_error_while_thinking_falls_back_to_a_direct_answer(self):
        from mathagent.agent import AgentError
        client = FakeClient()
        original = client.stream

        def stream(payload):
            if payload['think']:
                raise AgentError('Model stream error: Internal server error')
            yield from original(payload)
        client.stream = stream
        runner = QuickReviewRunner(self.agent(client))
        result = runner.start(WRONG, max_rounds=3)
        self.assertEqual(runner.state['verdict'], 'issues_found')
        self.assertTrue(any('failed on a check call with thinking' in w for w in runner.state['warnings']))
        self.assertTrue(all(not r['think'] for r in client.requests))

    def test_low_effort_is_one_pass_without_recheck(self):
        client = FakeClient()
        runner = QuickReviewRunner(self.agent(client))
        result = runner.start(WRONG, max_rounds=1)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(runner.state['verdict'], 'issues_alleged')
        self.assertIn('not re-checked', result['report'])

    def test_no_issue_and_rejected_objection(self):
        good = WRONG.replace('is continuous, hence', 'is compact (Rellich), hence')
        client = FakeClient()
        runner = QuickReviewRunner(self.agent(client))
        result = runner.start(good, max_rounds=3)
        self.assertEqual(runner.state['verdict'], 'no_issue_found')
        self.assertIn('No issue found** by 2 independent passes', result['report'])
        reject = {'confirm': lambda p, s: {'assessment': 'rejected', 'kind': 'invalid_inference',
                                           'argument': 'The step is fine.', 'suggestion': ''}}
        runner = QuickReviewRunner(self.agent(FakeClient(reject)))
        result = runner.start(WRONG, max_rounds=3)
        self.assertEqual(runner.state['verdict'], 'no_issue_found')
        self.assertIn('Objections raised and rejected', result['report'])
        self.assertIn('rejected by an independent re-check', result['report'])

    def test_quote_gate_relocates_or_discards(self):
        def wrong_line(prompt, schema):
            value = default_check(prompt, schema)
            for issue in value['issues']:
                issue['start_line'] = issue['end_line'] = 1  # outside the proof; the quote is on L4
            return value

        runner = QuickReviewRunner(self.agent(FakeClient({'check': wrong_line})))
        runner.start(WRONG, max_rounds=3)
        self.assertEqual(runner.state['verdict'], 'issues_found')
        self.assertTrue(all(i['start_line'] == 4 for c in runner.state['checks'].values() for i in c['issues']))

        def invented(prompt, schema):
            return {'explanation': 'x', 'verdict': 'issues_found', 'issues': [{
                'start_line': 3, 'end_line': 3, 'quote': 'by dominated convergence the integrals agree', 'kind': 'invalid_inference',
                'evidence': 'Invented objection.', 'suggestion': ''}]}

        reject = lambda p, s: {'assessment': 'rejected', 'kind': 'invalid_inference', 'argument': 'Not in the proof.', 'suggestion': ''}
        runner = QuickReviewRunner(self.agent(FakeClient({'check': invented, 'confirm': reject})))
        result = runner.start(WRONG, max_rounds=1)
        self.assertNotEqual(runner.state['verdict'], 'issues_found')
        self.assertIn('could not be located', result['report'])

    def test_statement_without_proof_is_refused(self):
        runner = QuickReviewRunner(self.agent(FakeClient()))
        result = runner.start('Lemma. Every bounded sequence in a Hilbert space has a weakly convergent subsequence.', max_rounds=1)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['worker_error']['code'], 'no_proof')

    def test_file_and_target_give_the_proof_with_its_context(self):
        (self.root / 'paper.tex').write_text(PAPER)
        client = FakeClient()
        runner = QuickReviewRunner(self.agent(client))
        result = runner.start('Is this proof right?', source_files=['paper.tex'], target='Theorem 1.1', max_rounds=1)
        self.assertEqual(runner.state['unit']['name'], 'Theorem 1.1')
        prompt = client.requests[0]['messages'][1]['content']
        self.assertIn('STATEMENTS OF RESULTS USED', prompt)
        self.assertIn('Definition 2.1', prompt)
        self.assertNotIn('Obvious.', prompt)  # other proofs stay out
        self.assertIn('[M1:L27]', result['report'])

    def test_concurrency_gives_the_same_report(self):
        drafts = []
        for concurrency in (1, 3):
            client = FakeClient(delay=0.01)
            runner = QuickReviewRunner(self.agent(client), concurrency=concurrency)
            runner.start(WRONG, max_rounds=10)
            self.assertEqual(len(runner.state['checks']), 5)
            self.assertLessEqual(runner.state['tokens_charged'], runner.state['settings']['max_tokens'])
            drafts.append(runner.state['draft'])
        self.assertEqual(drafts[0], drafts[1])

    def test_pause_and_resume_do_not_repeat_finished_passes(self):
        stop = lambda payload, count: count == 2
        client = FakeClient(fail_on=stop)
        runner = QuickReviewRunner(self.agent(client))
        result = runner.start(WRONG, max_rounds=3)
        self.assertEqual(result['status'], 'paused')
        self.assertEqual(len(runner.state['checks']), 1)
        client2 = FakeClient()
        saved = json.loads((Path(result['directory']) / 'state.json').read_text())
        self.assertIs(runner_for_state(saved), QuickReviewRunner)
        resumed = QuickReviewRunner(self.agent(client2)).resume(result['id'])
        self.assertEqual(resumed['status'], 'reviewed')
        checks = [r for r in client2.requests if 'LENS FOR THIS PASS' in r['messages'][1]['content']]
        self.assertEqual(len(checks), 1)


class ExplainTests(ReviewTestCase):
    PROOF = WRONG.replace('is continuous, hence', 'is compact (Rellich), hence')

    def explain(self, lines):
        def reply(prompt, schema):
            return {'says': 'A bounded sequence in H^1 has an L^2-convergent subsequence.', 'idea': 'Weak compactness, then Rellich.',
                    'steps': [{'start_line': a, 'end_line': b, 'what': f'Step {a}', 'why': 'Because.', 'uses': 'Hilbert space'} for a, b in lines],
                    'facts': [{'name': 'Rellich–Kondrachov theorem', 'statement': 'H^1(0,1) embeds compactly in L^2(0,1).'}],
                    'hypotheses': 'Boundedness of (0,1) is used by Rellich.', 'gaps': ''}
        return reply

    def test_every_line_is_explained_after_a_fill(self):
        fill = lambda p, s: {'steps': [{'start_line': 4, 'end_line': 5, 'what': 'Compactness gives strong convergence.', 'why': 'Rellich.', 'uses': 'Rellich'}]}
        client = FakeClient({'explain': self.explain([(3, 3)]), 'fill': fill})
        runner = ExplainRunner(self.agent(client))
        result = runner.start(self.PROOF, max_rounds=3)
        self.assertEqual(result['status'], 'reviewed')
        self.assertEqual([(s['start_line'], s['end_line']) for s in runner.state['explanation']['steps']], [(3, 3), (4, 5)])
        report = result['report']
        self.assertIn('Every line of the proof is explained.', report)
        self.assertIn('Rellich–Kondrachov theorem', report)
        self.assertIn('An independent pass found no problem', report)
        self.assertIn('> The embedding of H^1(0,1)', report)  # the source line next to its explanation

    def test_low_effort_has_no_accuracy_check_and_reports_missing_lines(self):
        fill = lambda p, s: {'steps': []}
        client = FakeClient({'explain': self.explain([(3, 3)]), 'fill': fill})
        runner = ExplainRunner(self.agent(client))
        result = runner.start(self.PROOF, max_rounds=1)
        self.assertNotIn(['problems'], self.kinds(client))
        self.assertIn('Lines not explained one by one: L4-L5.', result['report'])

    def test_problems_found_are_corrected(self):
        problems = lambda p, s: {'problems': [{'where': 'idea', 'problem': 'Rellich needs a bounded interval.', 'fix': 'Say so.'}]}
        client = FakeClient({'explain': self.explain([(3, 5)]), 'explain_check': problems})
        runner = ExplainRunner(self.agent(client))
        result = runner.start(self.PROOF, max_rounds=3)
        self.assertTrue(runner.state.get('explain_fixed'))
        self.assertIn('Problems found and corrected', result['report'])


class FakeLiterature:
    online = True
    deadline = None
    on_budget_change = None
    timeout = 5

    def __init__(self):
        self.calls = []
        self.stats = {'requests': 0, 'returned_chars': 0}

    def reset_budget(self, **kwargs):
        pass

    def snapshot(self):
        return {'online': self.online, 'stats': dict(self.stats)}

    def restore(self, value):
        pass

    def execute(self, name, args):
        self.calls.append((name, args))
        return json.dumps({'results': [{'title': 'Functional Analysis, Sobolev Spaces and PDE', 'authors': ['H. Brezis'],
                                        'year': 2011, 'abstract': 'A textbook.', 'url': 'https://example.org/brezis'}]})


class JournalTests(ReviewTestCase):
    def setUp(self):
        super().setUp()
        (self.root / 'paper.tex').write_text(PAPER)

    def test_report_from_checked_findings(self):
        client = FakeClient()
        runner = JournalRunner(self.agent(client))
        result = runner.start('Review this paper for a journal.', source_files=['paper.tex'], max_rounds=3)
        self.assertEqual(result['status'], 'reviewed')
        names = {u['id']: u['name'] for u in runner.state['map']['units']}
        self.assertEqual([names[u] for u in runner.state['plan_units']], ['Theorem 1.1', 'Lemma 2.2'])  # medium: no appendix
        report = result['report']
        self.assertIn('## Major comments', report)
        self.assertIn('1. *Theorem 1.1.* **Invalid inference** at [M1:L27]', report)
        self.assertIn('**Major revision**', report)
        self.assertNotIn('(provisional)', report)
        self.assertIn('| Theorem 1.1 (main) | L25-L28 | 1 | issues found |', report)
        self.assertIn('| Lemma A.1 | L34-L36 | 0 | not checked |', report)
        self.assertIn('Online search was off', report)
        # Understanding and presentation come first, read over the whole paper before any proof is checked.
        self.assertLess(report.index('## The paper in brief'), report.index('## Typos and presentation'))
        self.assertLess(report.index('## Typos and presentation'), report.index('## Major comments'))
        self.assertIn('[M1:L10] *definition* — "be the Sobolev space": Say which norm.', report)
        self.assertIn('Read for understanding and presentation:', report)
        roles = [r['messages'][1]['content'].split('\n')[0][:30] for r in client.requests]
        self.assertTrue(roles[0].startswith('Read this part'))
        # every check got the notation line the scope chose, before the statement
        checks = [r['messages'][1]['content'] for r in client.requests if 'LENS FOR THIS PASS' in r['messages'][1]['content']]
        self.assertTrue(all(c.index('L10: Let $H^1(0,1)$') < c.index('STATEMENT:') for c in checks))

    def test_the_review_reads_the_paper_but_checks_no_proof(self):
        client = FakeClient()
        runner = SummaryReviewRunner(self.agent(client))
        result = runner.start('Review.', source_files=['paper.tex'], max_rounds=3)
        self.assertEqual((result['status'], runner.state['variant']), ('reviewed', 'review'))
        self.assertFalse(runner.state['checks'])
        self.assertFalse(any('LENS FOR THIS PASS' in r['messages'][1]['content'] for r in client.requests))
        report = result['report']
        self.assertIn('## The paper in brief', report)
        self.assertIn('Say which norm.', report)
        self.assertIn('(the proofs were not checked)', report)
        self.assertNotIn('## Major comments', report)
        self.assertIn('did not check the proofs', client.requests[-1]['messages'][1]['content'])

    def test_small_budget_makes_the_recommendation_provisional(self):
        client = FakeClient()
        runner = JournalRunner(self.agent(client))
        with patch.object(JournalRunner, '_write_reserve', return_value=59000):
            result = runner.start('Review.', source_files=['paper.tex'], max_rounds=3, max_tokens=60000)
        self.assertEqual(result['status'], 'partial')
        self.assertIn('(provisional)', result['report'])
        self.assertIn('not checked', result['report'])

    def test_online_novelty_search_is_recorded(self):
        client = FakeClient()
        agent = self.agent(client)
        agent.workspace.literature = FakeLiterature()
        runner = JournalRunner(agent)
        with patch.object(JournalRunner, '_cited_checks', lambda self: None):
            result = runner.start('Review.', source_files=['paper.tex'], max_rounds=5)
        self.assertEqual(agent.workspace.literature.calls[0], ('search_papers', {'query': 'Rellich compactness Sobolev', 'limit': 5}))
        self.assertIn('Literature searched', result['report'])
        self.assertIn('[Functional Analysis, Sobolev Spaces and PDE](https://example.org/brezis)', result['report'])
        self.assertIn('textbook source', result['report'])
        self.assertIn('Say which norm.', result['report'])
        self.assertEqual(runner.state['evidence'][0]['tool'], 'search_papers')

    def test_queries_copied_from_the_manuscript_are_dropped(self):
        copied = lambda prompt, schema: {'field': 'f', 'contribution': 'c', 'main': [], 'notation': [], 'closest': [],
                                         'queries': ['Every bounded sequence in H^1(0,1) has a subsequence', 'Rellich compactness']}
        agent = self.agent(FakeClient({'scope': copied}))
        agent.workspace.literature = FakeLiterature()
        runner = JournalRunner(agent)
        with patch.object(JournalRunner, '_cited_checks', lambda self: None):
            runner.start('Review.', source_files=['paper.tex'], max_rounds=3)
        self.assertEqual(runner.state['scope']['queries'], ['Rellich compactness'])
        self.assertEqual([args['query'] for _, args in agent.workspace.literature.calls], ['Rellich compactness'])

    def test_guess_from_a_bibliography_entry(self):
        guess = JournalRunner._guess({'entry': 'H. Brezis. Functional Analysis, Sobolev Spaces and PDE. Springer, 2011.', 'place': 'Theorem 9.16'})
        self.assertEqual(guess[0]['authors'], ['H. Brezis'])
        self.assertEqual(guess[0]['title'], 'Functional Analysis, Sobolev Spaces and PDE')
        self.assertEqual((guess[0]['year'], guess[0]['locator']), ('2011', 'Theorem 9.16'))


class PdfCommentTests(unittest.TestCase):
    def test_only_comments_on_words_survive_a_pdf_extraction(self):
        word = {'quote': 'remainder terms in the theorem Theorem 1.4', 'comment': 'Theorem is repeated.', 'kind': 'typo'}
        formula = {'quote': 'b k = ∆Ψk · ∇Λ ◦ Ψk', 'comment': 'Inconsistent hat.', 'kind': 'typo'}
        blamed = {'quote': 'Notice that the function is smooth', 'comment': 'An extraction artifact splits it.', 'kind': 'typo'}
        notation = {'quote': 'outside the domain of the problem', 'comment': 'Use the same symbol.', 'kind': 'notation'}
        self.assertEqual([_prose_comment(c) for c in (word, formula, blamed, notation)], [True, False, False, False])


class LevelTests(unittest.TestCase):
    def test_levels_follow_effort_tries(self):
        self.assertEqual([level_for(n) for n in (1, 3, 5, 7, 10)], ['low', 'medium', 'high', 'xhigh', 'poincare'])
        self.assertEqual([EFFORT[level_for(n)]['passes'] for n in (1, 3, 5, 7, 10)], [1, 2, 3, 4, 5])
        self.assertIs(runner_for_state({'kind': 'referee'}), ResearchRunner)


if __name__ == '__main__':
    unittest.main()
