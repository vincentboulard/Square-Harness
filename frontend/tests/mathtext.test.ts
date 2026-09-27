// Run with: npm test  (Node 22.18+ strips the TypeScript types itself)
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { normalizeMath, snippet, splitBlocks } from '../src/mathtext.ts'

const cases: [string, string, string][] = [
  ['inline dollars are untouched', 'Let $a_1$ and $b_2$ be even.', 'Let $a_1$ and $b_2$ be even.'],
  ['parenthesis delimiters', 'Let \\(x_n \\to 0\\) hold.', 'Let $x_n \\to 0$ hold.'],
  ['bracket delimiters on their own lines', 'Then\n\\[\n a = 2m \n\\]\nso', 'Then\n$$\na = 2m\n$$\nso'],
  ['one-line display becomes a block', 'We get\n$$ a = n_1 \\cdot 2 $$\nhence', 'We get\n$$\na = n_1 \\cdot 2\n$$\nhence'],
  ['display inside a sentence stays inline', 'Derivation: $$a = 2m$$ implies it.', 'Derivation: $\\displaystyle a = 2m$ implies it.'],
  ['display keeps list indentation', '* item\n    $$x^2$$\n* next', '* item\n    $$\n    x^2\n    $$\n* next'],
  ['bare align environment', 'So\n\\begin{align*}\na &= b\n\\end{align*}\nDone', 'So\n$$\n\\begin{align*}\na &= b\n\\end{align*}\n$$\nDone'],
  ['environment already in display math', '$$\n\\begin{aligned} a&=b \\end{aligned}\n$$', '$$\n\\begin{aligned} a&=b \\end{aligned}\n$$'],
  ['text-mode bold outside math only', '\\textbf{Step 1.} Use $\\textbf{x}$.', '**Step 1.** Use $\\textbf{x}$.'],
  ['proof environment ends with a tombstone', '\\begin{proof}Trivial.\\end{proof}', '*Proof.* Trivial. ∎'],
  ['theorem environment with a title', '\\begin{theorem}[Rellich] Compact.\\end{theorem}', '**Theorem (Rellich).**  Compact.'],
  ['code is never rewritten', 'Run `$x$ \\(y\\)` and\n```\n\\[ z \\]\n```', 'Run `$x$ \\(y\\)` and\n```\n\\[ z \\]\n```'],
  ['labels dropped, references kept', 'By \\eqref{eq:1}\\label{x} done', 'By (eq:1) done'],
]

for (const [name, input, expected] of cases) {
  test(name, () => assert.equal(normalizeMath(input), expected))
}

test('streaming blocks never split display math or code fences', () => {
  assert.equal(splitBlocks('para one\n\n$$\na\n\nb\n$$\n\n```\nx\n\ny\n```\n\nend').length, 4)
})

test('snippets never end inside inline math', () => {
  assert.equal(snippet('The hypothesis states that $a$ and $b$ are even integers, which implies $a = 2m$', 40), 'The hypothesis states that $a$ and $b$…')
  assert.equal(snippet('Cut here $x + y = z$ inside the formula region please', 14), 'Cut here…')
  assert.equal(snippet('Short text', 40), 'Short text')
})
