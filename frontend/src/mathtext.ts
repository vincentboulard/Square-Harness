// Pure text preparation for typeset mathematics (no React; testable with node).

const CODE = /(```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`\n]+`)/g
const MATH = /\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g
const DISPLAY_ENV = /(^|\n)([ \t]*)(\\begin\{(equation|align|gather|multline|flalign|alignat)(\*?)\}[\s\S]*?\\end\{\4\5\})/g
const THEOREM = 'theorem|lemma|proposition|corollary|definition|remark|example|claim|conjecture|assumption|hypothesis'

const capitalize = (word: string) => word.charAt(0).toUpperCase() + word.slice(1)

/** Text-mode LaTeX that models often emit in prose, outside math. */
function prose(text: string): string {
  return text
    .replace(/\\textbf\{([^{}\n]*)\}/g, '**$1**')
    .replace(/\\(?:emph|textit)\{([^{}\n]*)\}/g, '*$1*')
    .replace(new RegExp(`\\\\begin\\{(${THEOREM})\\}(?:\\[([^\\]\\n]*)\\])?`, 'g'),
      (_m, env: string, title?: string) => `**${capitalize(env)}${title ? ` (${title})` : ''}.** `)
    .replace(new RegExp(`\\\\end\\{(${THEOREM})\\}`, 'g'), '')
    .replace(/\\begin\{proof\}(?:\[([^\]\n]*)\])?/g, (_m, title?: string) => `*${title || 'Proof'}.* `)
    .replace(/\\end\{proof\}/g, ' ∎')
    .replace(/\\label\{[^{}\n]*\}/g, '')
    .replace(/\\(?:eq)?ref\{([^{}\n]*)\}/g, '($1)')
}

function normalizeText(text: string): string {
  text = text
    .replace(/\\\[([\s\S]+?)\\\]/g, (_m, body: string) => `$$${body}$$`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_m, body: string) => `$${body.trim()}$`)
  // A bare display environment on its own lines becomes display math.
  text = text.replace(DISPLAY_ENV, (match, start: string, indent: string, env: string, _n: string, _s: string, offset: number, whole: string) => {
    const before = whole.slice(0, offset + start.length).trimEnd()
    return before.endsWith('$$') ? match : `${start}${indent}$$\n${env}\n${indent}$$`
  })
  let out = ''
  let last = 0
  for (const match of text.matchAll(MATH)) {
    const at = match.index ?? 0
    out += prose(text.slice(last, at))
    if (match[1] !== undefined) {
      const lineStart = text.lastIndexOf('\n', at - 1) + 1
      const lineEnd = text.indexOf('\n', at + match[0].length)
      const before = text.slice(lineStart, at)
      const after = text.slice(at + match[0].length, lineEnd === -1 ? text.length : lineEnd)
      const body = match[1].trim()
      if (!before.trim() && !after.trim()) {
        // Alone on its line(s): a display block, keeping list indentation.
        out += `$$\n${before}${body.split('\n').map((line) => line.trim()).join('\n' + before)}\n${before}$$`
      } else {
        // Inside a sentence: keep the paragraph intact, typeset in display style.
        out += `$\\displaystyle ${body.replace(/\s*\n\s*/g, ' ')}$`
      }
    } else {
      out += match[0]
    }
    last = at + match[0].length
  }
  return out + prose(text.slice(last))
}

export function normalizeMath(source: string): string {
  return source.split(CODE).map((part, index) => (index % 2 ? part : normalizeText(part))).join('')
}

// -- citations such as [M1:L3-L6] or [doc-0123abcd:L12-L28] -----------------------

const CITE = /\[((?:doc-[A-Za-z0-9_-]+|S[\w-]+|M\d+)(?::L\d+(?:-L?\d+)?)?)\]/g
type MdNode = { type: string; value?: string; children?: MdNode[]; data?: Record<string, unknown> }

function splitCitations(node: MdNode): MdNode[] {
  const value = node.value || ''
  const out: MdNode[] = []
  let last = 0
  for (const match of value.matchAll(CITE)) {
    const at = match.index ?? 0
    if (at > last) out.push({ type: 'text', value: value.slice(last, at) })
    out.push({ type: 'citation', data: { hName: 'cite', hProperties: { dataCite: match[1] } },
      children: [{ type: 'text', value: match[1] }] })
    last = at + match[0].length
  }
  if (!out.length) return [node]
  if (last < value.length) out.push({ type: 'text', value: value.slice(last) })
  return out
}

function walk(node: MdNode) {
  if (!node.children) return
  node.children = node.children.flatMap((child) => {
    if (child.type === 'text') return splitCitations(child)
    if (!['code', 'inlineCode', 'math', 'inlineMath', 'link'].includes(child.type)) walk(child)
    return [child]
  })
}

export const remarkCitations = () => (tree: MdNode) => walk(tree)

/** Split streaming text into stable blocks so only the growing tail re-renders. */
export function splitBlocks(text: string): string[] {
  const blocks: string[] = []
  let current: string[] = []
  let fence = false
  let math = false
  for (const line of text.split('\n')) {
    const trimmed = line.trim()
    if (/^(```|~~~)/.test(trimmed)) fence = !fence
    else if (!fence && trimmed === '$$') math = !math
    if (!trimmed && !fence && !math && current.length) {
      blocks.push(current.join('\n'))
      current = []
    } else {
      current.push(line)
    }
  }
  if (current.length) blocks.push(current.join('\n'))
  return blocks
}

/** Shorten text for a one-line snippet without cutting inside inline math. */
export function snippet(text: string, limit: number): string {
  const flat = text.replace(/\s+/g, ' ').trim()
  if (flat.length <= limit) return flat
  let cut = flat.lastIndexOf(' ', limit)
  if (cut < limit * 0.6) cut = limit
  let head = flat.slice(0, cut)
  if ((head.match(/\$\$/g) || []).length % 2 === 1) head = head.slice(0, head.lastIndexOf('$$'))
  if ((head.replace(/\$\$/g, '').match(/(?<!\\)\$/g) || []).length % 2 === 1) head = head.slice(0, head.lastIndexOf('$'))
  return head.trimEnd() + '…'
}
