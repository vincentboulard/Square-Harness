"""The manuscript map: statements, proofs, dependencies and citations, without any model."""
import unittest

from mathagent import manuscript

TEX = r"""\documentclass{amsart}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem*{claim}{Claim}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\begin{document}
\begin{abstract}
We prove a compactness result.
\end{abstract}
\section{Introduction}
\begin{theorem}[Main result]\label{thm:main}
Every bounded sequence in $H^1(0,1)$ has a subsequence converging in $L^2(0,1)$.
\end{theorem}
\section{Preliminaries}
\begin{definition}\label{def:h1}
$H^1(0,1)$ is the space of $L^2$ functions with $L^2$ weak derivative.
\end{definition}
\begin{lemma}\label{lem:embed}
The embedding $H^1(0,1)\hookrightarrow L^2(0,1)$ is compact.
\end{lemma}
\begin{proof}
This is Rellich's theorem, see \cite[Theorem 9.16]{brezis}. % a comment \ref{thm:main}
By \eqref{eq:bound} the sequence is bounded.
\end{proof}
\begin{equation}\label{eq:bound}
\|u\|_{L^2} \le \|u\|_{H^1}
\end{equation}
\section{Proof of the main result}
\begin{proof}[Proof of Theorem~\ref{thm:main}]
By Lemma~\ref{lem:embed} and Definition~\ref{def:h1}, the claim follows.
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
\bibitem{brezis} H. Brezis, Functional Analysis, Springer, 2011.
\end{thebibliography}
\end{document}
"""

MARKDOWN = """# Notes on compactness

## 1. Setting

**Definition 1.1.** A sequence is bounded if its norms are bounded.

**Lemma 1.2** (Rellich). The embedding is compact.

*Proof.* See Theorem 9.16 of Brezis. ∎

**Theorem 1.3.** Every bounded sequence has a convergent subsequence.

Proof. Lemma 1.2 applied to Definition 1.1 gives the result.
"""


class TexMapTests(unittest.TestCase):
    def setUp(self):
        self.map = manuscript.build(TEX, 'paper.tex')
        self.by_name = {u['name']: u for u in self.map['units']}

    def test_numbers_follow_shared_counters_and_sections(self):
        self.assertEqual(self.map['format'], 'tex')
        self.assertEqual(list(self.by_name), ['Theorem 1.1', 'Definition 2.1', 'Lemma 2.2', 'Lemma A.1'])
        self.assertEqual(self.map['abstract'], [8, 10])
        self.assertTrue(self.by_name['Lemma A.1']['appendix'])
        self.assertFalse(self.by_name['Lemma 2.2']['appendix'])

    def test_proofs_attach_to_their_statements(self):
        main = self.by_name['Theorem 1.1']
        self.assertTrue(main['main'])
        self.assertEqual(main['title'], 'Main result')
        self.assertEqual(main['proofs'], [[30, 32]])
        self.assertEqual(self.by_name['Lemma 2.2']['proofs'], [[22, 25]])
        self.assertEqual(self.by_name['Definition 2.1']['proofs'], [])

    def test_dependencies_citations_and_equations(self):
        lemma, main = self.by_name['Lemma 2.2'], self.by_name['Theorem 1.1']
        self.assertEqual(main['refs'], [lemma['id'], self.by_name['Definition 2.1']['id']])
        self.assertEqual(lemma['refs'], [])  # the commented \ref is ignored
        self.assertEqual(lemma['cites'], [{'key': 'brezis', 'note': 'Theorem 9.16', 'line': 23}])
        self.assertEqual(lemma['equations'], [[26, 28]])
        self.assertIn('Brezis', self.map['bibliography']['brezis'])

    def test_resolve_and_priority(self):
        self.assertEqual(manuscript.resolve(self.map, 'Lemma 2.2')['label'], 'lem:embed')
        self.assertEqual(manuscript.resolve(self.map, 'check lem:embed please')['name'], 'Lemma 2.2')
        self.assertEqual(manuscript.resolve(self.map, 'the main theorem')['name'], 'Theorem 1.1')
        self.assertIsNone(manuscript.resolve(self.map, 'Lemma 7.4'))
        ids = {u['name']: u['id'] for u in self.map['units']}
        self.assertEqual(manuscript.priority(self.map, depth=0), [ids['Theorem 1.1']])
        self.assertEqual(manuscript.priority(self.map, depth=1), [ids['Theorem 1.1'], ids['Lemma 2.2']])
        self.assertEqual(manuscript.priority(self.map, appendix=False), [ids['Theorem 1.1'], ids['Lemma 2.2']])
        self.assertEqual(manuscript.priority(self.map), [ids['Theorem 1.1'], ids['Lemma 2.2'], ids['Lemma A.1']])

    def test_without_declarations_labels_name_the_statements(self):
        mapping = manuscript.build('\\begin{lemma}\\label{lem:x}\nA.\n\\end{lemma}\n\\begin{proof}\nB.\n\\end{proof}\n', 'n.tex')
        self.assertEqual(mapping['units'][0]['name'], 'Lemma (lem:x)')
        self.assertEqual(mapping['units'][0]['proofs'], [[4, 6]])
        self.assertTrue(mapping['units'][0]['main'])
        self.assertTrue(mapping['warnings'])


class TextMapTests(unittest.TestCase):
    def test_markdown_statements_and_bare_proofs(self):
        mapping = manuscript.build(MARKDOWN, 'notes.md')
        names = [u['name'] for u in mapping['units']]
        self.assertEqual(names, ['Definition 1.1', 'Lemma 1.2', 'Theorem 1.3'])
        lemma, theorem = mapping['units'][1], mapping['units'][2]
        self.assertEqual(lemma['title'], 'Rellich')
        self.assertEqual(lemma['proofs'], [[9, 9]])
        self.assertEqual(theorem['proofs'], [[13, 13]])
        self.assertEqual(theorem['refs'], [lemma['id'], mapping['units'][0]['id']])
        self.assertTrue(theorem['main'])

    def test_pdf_like_text(self):
        lines = ['1. Introduction', 'Theorem 1.1. The main claim holds.', 'Theorem 1.1 is proved below.',
                 '2. Proofs', 'Lemma 2.1. A lemma.', 'Proof. Easy, see [2, Theorem 3].     □',
                 'Proof of Theorem 1.1. By Lemma 2.1 and (2.3) we conclude.', '(2.3)    a = b', 'end.   □',
                 'References', '[1] A. Author. A book. 2000.', '[2] B. Author. A paper. 2001.']
        mapping = manuscript.build('\n'.join(lines), 'paper.pdf')
        theorem, lemma = mapping['units']
        self.assertEqual((theorem['name'], lemma['name']), ('Theorem 1.1', 'Lemma 2.1'))
        self.assertEqual(theorem['proofs'], [[7, 9]])
        self.assertEqual(theorem['refs'], [lemma['id']])
        self.assertEqual(lemma['cites'], [{'key': '2', 'note': 'Theorem 3', 'line': 6}])
        self.assertEqual(sorted(mapping['bibliography']), ['1', '2'])
        self.assertEqual([s['number'] for s in mapping['sections']], ['1', '2'])

    def test_sentences_starting_with_a_reference_are_not_statements(self):
        mapping = manuscript.build('Lemma 2.1. A lemma.\nLemma 2.1 makes this explicit.\nProposition 3 for details.', 'x.txt')
        self.assertEqual([u['name'] for u in mapping['units']], ['Lemma 2.1'])

    def test_unstructured_text_warns(self):
        mapping = manuscript.build('just some notes\nwithout statements', 'x.md')
        self.assertEqual(mapping['units'], [])
        self.assertTrue(mapping['warnings'])
        self.assertEqual(manuscript.chunks(250, 120), [[1, 120], [121, 240], [241, 250]])


if __name__ == '__main__':
    unittest.main()
