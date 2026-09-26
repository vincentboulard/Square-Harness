export function ago(iso: string | number | null | undefined): string {
  if (!iso) return ''
  const time = typeof iso === 'number' ? iso * 1000 : Date.parse(iso)
  const seconds = Math.max(0, (Date.now() - time) / 1000)
  if (seconds < 45) return 'just now'
  if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`
  if (seconds < 86400) return `${Math.round(seconds / 3600)} h ago`
  const days = Math.round(seconds / 86400)
  if (days < 7) return `${days} d ago`
  return new Date(time).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
}

export function duration(seconds: number): string {
  if (!isFinite(seconds)) return ''
  if (seconds < 60) return `${Math.round(seconds)} s`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes} min ${Math.round(seconds % 60)} s`
  return `${Math.floor(minutes / 60)} h ${minutes % 60} min`
}

export function count(value: number): string {
  return new Intl.NumberFormat(undefined).format(Math.round(value))
}

export function bytes(value: number): string {
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / 1024 / 1024).toFixed(1)} MB`
}

export function plural(n: number, one: string, many = one + 's') {
  return `${count(n)} ${n === 1 ? one : many}`
}

export function firstLine(text: string, limit = 120): string {
  const line = text.replace(/\s+/g, ' ').trim()
  return line.length <= limit ? line : line.slice(0, limit - 1) + '…'
}
