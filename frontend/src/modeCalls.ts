import type { RouteMode } from './api'
export const MODE_CALLS: { mode: RouteMode; label: string; summary: string }[] = [
  { mode: 'prove', label: 'Prove', summary: 'Prove a statement, review the proof, and repair objections.' },
  { mode: 'check', label: 'Reference check', summary: 'Find or verify a reference for a known result.' },
  { mode: 'literature', label: 'Literature', summary: 'Build a verified reading list on a topic.' },
  { mode: 'referee', label: 'Review', summary: 'Review a manuscript and report precise issues.' },
]
export type ModeCall = { mode: RouteMode; start: number; end: number }
export type ModeQuery = { start: number; end: number; query: string }
const word = (char: string) => /[\p{L}\p{N}_@.+/\-]/u.test(char)

export function validCall(text: string, call: ModeCall | null): call is ModeCall {
  return !!call && text.slice(call.start, call.end) === '@' + call.mode
    && (!call.start || !word(text[call.start - 1]))
    && (call.end === text.length || !word(text[call.end]))
}

export function modeQuery(text: string, caret: number, end = caret): ModeQuery | null {
  if (caret !== end) return null
  const before = text.slice(0, caret)
  const match = /@([a-z]*)$/i.exec(before)
  if (!match || (match.index && word(text[match.index - 1]))) return null
  const tail = /^[a-z]*/i.exec(text.slice(caret))![0]
  const tokenEnd = caret + tail.length
  if (tokenEnd < text.length && word(text[tokenEnd])) return null
  return { start: match.index, end: tokenEnd, query: match[1].toLowerCase() }
}

export function updateCall(before: string, after: string, call: ModeCall | null): ModeCall | null {
  if (!call) return null
  let start = 0
  while (start < before.length && start < after.length && before[start] === after[start]) start++
  let end = before.length, nextEnd = after.length
  while (end > start && nextEnd > start && before[end - 1] === after[nextEnd - 1]) { end--; nextEnd-- }
  const delta = after.length - before.length
  const next = end <= call.start ? { ...call, start: call.start + delta, end: call.end + delta }
    : start >= call.end ? call : null
  return validCall(after, next) ? next : null
}

export function callPayload(text: string, call: ModeCall | null) {
  // DOM selections use UTF-16 offsets; the server indexes Unicode code points.
  return validCall(text, call) ? { mode: call.mode, start: Array.from(text.slice(0, call.start)).length } : undefined
}
