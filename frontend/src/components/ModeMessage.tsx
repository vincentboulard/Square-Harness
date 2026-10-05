import { useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent } from 'react'
import { MODE_CALLS, modeQuery, updateCall, validCall, type ModeCall, type ModeQuery } from '../modeCalls'
import { CloseIcon } from './Icons'
import { Glyph } from './Glyph'
import './ModeMessage.css'

export function ModeMessage({ value, onChange, call, onCall, onSend, disabled }: {
  value: string; onChange: (value: string) => void
  call: ModeCall | null; onCall: (call: ModeCall | null) => void
  onSend: () => void; disabled: boolean
}) {
  const input = useRef<HTMLTextAreaElement>(null)
  const selectedCaret = useRef<number | null>(null)
  const listId = useId()
  const [query, setQuery] = useState<ModeQuery | null>(null)
  const queryRef = useRef<ModeQuery | null>(null)
  const dismissed = useRef<{ text: string; start: number } | null>(null)
  const [active, setActive] = useState(0)
  const options = query ? MODE_CALLS.filter((item) => item.mode.startsWith(query.query)) : []
  const open = !disabled && options.length > 0
  useLayoutEffect(() => {
    const caret = selectedCaret.current
    if (caret === null) return
    selectedCaret.current = null
    input.current?.focus()
    input.current?.setSelectionRange(caret, caret)
  }, [value, call])
  useEffect(() => {
    if (open) document.getElementById(`${listId}-${active}`)?.scrollIntoView({ block: 'nearest' })
  }, [active, open, listId])
  const refresh = (text: string, start: number, end: number) => {
    const next = modeQuery(text, start, end)
    const shown = next && !(validCall(text, call) && next.start === call.start)
      && !(dismissed.current?.text === text && dismissed.current.start === next.start) ? next : null
    const previous = queryRef.current
    if (shown?.start === previous?.start && shown?.end === previous?.end && shown?.query === previous?.query) return
    queryRef.current = shown
    setQuery(shown)
    setActive(0)
  }
  const select = (index: number) => {
    if (!query || !options[index]) return
    const mode = options[index].mode
    const token = '@' + mode
    const suffix = value.slice(query.end)
    const space = suffix.length === 0 ? ' ' : ''
    const next = value.slice(0, query.start) + token + space + suffix
    selectedCaret.current = query.start + token.length + space.length
    onChange(next)
    onCall({ mode, start: query.start, end: query.start + token.length })
    setQuery(null)
    queryRef.current = null
  }
  const keys = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.nativeEvent.isComposing) return
    if (open && !event.shiftKey && !event.ctrlKey && !event.metaKey && !event.altKey) {
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault()
        setActive((index) => (index + (event.key === 'ArrowDown' ? 1 : -1) + options.length) % options.length)
        return
      }
      if (event.key === 'Enter' || event.key === 'Tab') { event.preventDefault(); select(active); return }
      if (event.key === 'Escape') {
        event.preventDefault()
        dismissed.current = { text: value, start: query!.start }
        queryRef.current = null
        setQuery(null)
        return
      }
    }
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); onSend() }
  }
  return (
    <div className="mode-message">
      {validCall(value, call) && <div className="selected-mode" role="status">
        <span className={`route-cover mode-${call.mode}`}><Glyph mode={call.mode} /></span>
        <strong>@{call.mode}</strong><span>Selected for this message</span>
        <button type="button" className="chip-remove" aria-label="Clear selected mode" onClick={() => { onCall(null); setQuery(null); input.current?.focus() }}><CloseIcon size={13} /></button>
      </div>}
      {open && <div className="mode-picker">
        <div className="mode-picker-title">Modes <span>↑ ↓ to browse · Enter or Tab to select · Esc to dismiss</span></div>
        <div id={listId} role="listbox" aria-label="Choose a mode">
          {options.map((item, index) => <div key={item.mode} id={`${listId}-${index}`} role="option"
            aria-selected={index === active} className={'mode-option' + (index === active ? ' mode-option-active' : '')}
            onMouseEnter={() => setActive(index)} onMouseDown={(event) => event.preventDefault()} onClick={() => select(index)}>
            <span className={`route-cover mode-${item.mode}`}><Glyph mode={item.mode} /></span>
            <span><strong>@{item.mode}</strong><span className="mode-option-summary">{item.summary}</span></span>
          </div>)}
        </div>
      </div>}
      <textarea ref={input} value={value} onChange={(event) => {
        const next = event.target.value
        onCall(updateCall(value, next, call))
        onChange(next)
        refresh(next, event.target.selectionStart, event.target.selectionEnd)
      }} onSelect={(event) => refresh(value, event.currentTarget.selectionStart, event.currentTarget.selectionEnd)}
        onBlur={() => { queryRef.current = null; setQuery(null) }} onKeyDown={keys}
        rows={Math.min(8, Math.max(2, value.split('\n').length))}
        placeholder="Describe what you want to do; type @ to select a mode…" aria-label="Message"
        aria-autocomplete="list" aria-expanded={open} aria-controls={open ? listId : undefined}
        aria-activedescendant={open ? `${listId}-${active}` : undefined} />
    </div>
  )
}
