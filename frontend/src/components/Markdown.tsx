// Markdown with typeset mathematics. Model and source text is untrusted:
// no raw HTML, no images, KaTeX with trust disabled, and only http(s)/mailto links.
import { memo, useMemo } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { normalizeMath, remarkCitations, snippet, splitBlocks } from '../mathtext'

const KATEX = { throwOnError: false, strict: 'ignore', trust: false, maxExpand: 500, maxSize: 20 } as const

// -- rendering ---------------------------------------------------------------------

const safeUrl = (url: string) => (/^(https?:|mailto:)/i.test(url) ? url : '')

type Props = {
  children: string
  className?: string
  onCite?: (ref: string) => void
  flagged?: Set<string>
  inline?: boolean
}

export const Markdown = memo(function Markdown({ children, className, onCite, flagged, inline }: Props) {
  const components = useMemo<Components>(() => ({
    a: ({ href, children: label }) => (
      <a href={href} target="_blank" rel="noopener noreferrer nofollow">{label}</a>
    ),
    img: ({ alt }) => <span className="img-alt">(image{alt ? `: ${alt}` : ''} not loaded)</span>,
    ...(inline ? { p: ({ children: text }) => <>{text}</> } : {}),
    cite: (props) => {
      const ref = String((props as Record<string, unknown>)['data-cite'] || '')
      const bad = flagged?.has(ref)
      return (
        <button type="button" className={'cite' + (bad ? ' cite-flagged' : '')}
          title={bad ? 'The controller flagged this citation' : 'Show the cited passage'}
          onClick={() => onCite?.(ref)}>
          {ref}
        </button>
      )
    },
  }), [onCite, flagged, inline])
  const source = useMemo(() => normalizeMath(children || ''), [children])
  const Wrapper = inline ? 'span' : 'div'
  return (
    <Wrapper className={(inline ? 'prose-inline' : 'prose') + (className ? ' ' + className : '')}>
      <ReactMarkdown
        remarkPlugins={onCite ? [remarkGfm, remarkMath, remarkCitations] : [remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, KATEX]]}
        components={components}
        urlTransform={safeUrl}
      >
        {source}
      </ReactMarkdown>
    </Wrapper>
  )
})

/** One line of text with its mathematics typeset, shortened safely. */
export function Inline({ children, limit, className }: { children: string; limit?: number; className?: string }) {
  const text = limit ? snippet(children, limit) : children.replace(/\s+/g, ' ').trim()
  return <Markdown inline className={className}>{text}</Markdown>
}

export function StreamingMarkdown({ text, className }: { text: string; className?: string }) {
  const blocks = splitBlocks(text)
  return (
    <div className={'streaming' + (className ? ' ' + className : '')}>
      {blocks.map((block, index) => <Markdown key={index}>{block}</Markdown>)}
    </div>
  )
}
