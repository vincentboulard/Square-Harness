// Typed client for the harness's local /api. Shapes mirror mathagent/gui/store.py.

export type Budget = { used: number; limit: number }

export type Defaults = {
  ctx: number
  predict: number
  think: boolean
  proof_rounds: number
  proof_tokens: number
  proof_seconds: number
  proof_solve_tokens: number
  proof_verify_tokens: number
  proof_repair_tokens: number | null
  proof_min_solve_tokens: number | null
  research_rounds: number
  research_tokens: number
  research_input_tokens: number
  research_seconds: number
  research_requests: number
  research_chars: number
}

export type Status = {
  version: string
  instance: string
  seq: number
  workspace: string
  model: string
  host: string
  backend: 'ollama' | 'openai' | 'llamacpp'
  online: boolean
  online_locked: boolean
  allow_python: boolean
  lan: boolean
  defaults: Defaults
  ollama: { reachable: boolean; models: string[]; error?: string }
  pair_urls?: string[]
  root: string
  workspace_path: string
  read: ReadSettings
  latex: boolean
  recent: string[]
}

export type ReadSettings = { tex: boolean; pdf: boolean; py: boolean; text: boolean }

export type FolderListing = {
  root: string
  path: string
  current: string
  recent: string[]
  files: { tex: number; pdf: number; py: number }
  folders: { name: string; path: string; files: { tex: number; pdf: number; py: number }; jobs: boolean }[]
}

export type Template = { name: string; files: string[] }

export type TaskState = 'queued' | 'starting' | 'running' | 'pausing' | 'paused' | 'done' | 'error' | 'cancelled'

export type TaskSummary = {
  id: string
  kind: 'proof' | 'research' | 'chat'
  target: string | null
  label: string
  title: string
  state: TaskState
  error: string | null
  result_status: string | null
  started: number
  finished: number | null
}

export type Approval = {
  id: string
  task: string
  task_kind: string
  target: string | null
  kind: 'write' | 'python' | 'confirm'
  preview: string
}

export type LiveStep =
  | { type: 'call'; thinking: string; text: string }
  | { type: 'tool' | 'result' | 'notice'; text: string }

export type LiveTurn = { chat: string; kind: 'message' | 'review'; user: string; steps: LiveStep[]; mode?: 'critic' | 'explore' | 'check' }

export type Activity = { kind: 'notice' | 'tool' | 'result'; text: string; time: number }

export type TaskSnapshot = {
  seq: number
  task: TaskSummary | null
  activity: Activity[]
  live: LiveTurn | null
  approvals: Approval[]
  queue: TaskSummary[]
  routing: string[]
}

// Proof jobs (v0.5 engine): immutable candidates, whole-proof reviews, recorded calls.
export type ReviewStatus = 'unreviewed' | 'no_issue_found' | 'issues_found' | 'uncertain' | 'review_unavailable'

export type Candidate = {
  id: string
  attempt: number
  call: string
  artifact: string
  sha256: string
  parent: string | null
  kind: 'initial' | 'repair' | 'continue' | 'retry'
  transport_complete: boolean
  review_status: ReviewStatus
}

export type ReviewIssue = { kind: 'invalid_inference' | 'missing_justification' | 'uncertainty'; location: string; evidence: string }

export type Review = {
  id: string
  candidate: string
  candidate_sha256: string
  call: string
  response: { verdict: 'no_issue_found' | 'issues_found' | 'uncertain'; explanation: string; issues: ReviewIssue[] } | null
  protocol_error: string | null
}

export type Pending = {
  phase: 'solve' | 'review' | 'continue' | 'repair'
  kind: Candidate['kind']
  parent: string | null
  review: string | null
  index: number
  review_attempt: number
  candidate?: string
  solver_cap?: number
  continuation_fallback?: string
  notes_chars_kept?: number
  notes_chars_total?: number
}

export type Selection = { previous: string | null; selected: string; reason: string; after_call_count: number }

export type Call = {
  key?: string
  role: string
  round: number
  status: string
  reserved_tokens: number
  charged_tokens?: number
  stream: string
  request: string
  seconds?: number
  stats?: { eval_count?: number; prompt_eval_count?: number; done_reason?: string }
  context?: { input_tokens: number; method: string }
}

export type Live = { file: string; role: string; round?: number } | null

export type ProofListItem = {
  id: string
  kind: 'proof'
  version: number
  status: string
  title: string
  updated_at: string
  created_at: string
  running: boolean
  sources: string[]
  rounds: Budget
  tokens: Budget
  reviews: Partial<Record<ReviewStatus, number>>
}

export type ProofDetail = {
  id: string
  kind: 'proof'
  version: number
  // Made by the v0.4 engine: readable, never resumed.
  legacy: boolean
  goal: string
  status: string
  stop_reason: string | null
  recovery_notice: string | null
  created_at: string
  updated_at: string
  revision: number
  settings: Record<string, unknown> & {
    max_rounds: number; model: string; ctx: number; max_predict?: number; verify_tokens?: number
    repair_tokens?: number; verify_temperature?: number; harness_version?: string
  }
  budget: { rounds: Budget; tokens: Budget; seconds: Budget }
  sources: { path: string; sha256?: string; lines: number }[]
  candidates: Candidate[]
  reviews: Review[]
  calls: Call[]
  pending: Pending | null
  selected_candidate: string | null
  initial_candidate: string | null
  selection_history: Selection[]
  // The selected candidate's exact text (for a legacy job, its audited proof.md).
  answer: string | null
  running: boolean
  live: Live
  artifacts: { name: string; size: number }[]
}

export type Evidence = {
  id: string
  tool: string
  arguments: Record<string, unknown>
  artifact: string
  result: string
  round: number
}

export type ResearchListItem = {
  id: string
  kind: 'literature' | 'referee' | 'writeup'
  status: string
  phase: string
  title: string
  updated_at: string
  created_at: string
  running: boolean
  sources: string[]
  tokens: Budget
}

export type WriteupSection = { title: string; sources: string[]; goal: string; latex: string; complete: boolean; truncated: boolean }

export type CompileResult = {
  available: boolean
  ok: boolean
  errors: { file: string | null; line: number | null; message: string }[]
  log: string | null
  failing_sections: number[]
  pdf: boolean
}

export type ResearchDetail = {
  id: string
  kind: 'literature' | 'referee' | 'writeup'
  goal: string
  status: string
  phase: string
  stop_reason: string | null
  created_at: string
  updated_at: string
  settings: Record<string, unknown> & { online?: boolean; model: string }
  budget: { rounds: Budget; tokens: Budget; input_tokens: Budget; seconds: Budget; requests: Budget; chars: Budget }
  sources: { id: string; path: string; sha256: string; lines: number }[]
  plan: string
  notes: { round: number; title?: string; text: string; complete: boolean }[]
  pipeline?: string | null
  evidence: Evidence[]
  draft: string
  review: string
  draft_complete?: boolean
  review_complete?: boolean
  warnings: string[]
  citation_issues: string[]
  review_context_issues: string[]
  manuscript_ranges: Record<string, [number, number][]>
  calls: Call[]
  report: string
  running: boolean
  live: Live
  artifacts: { name: string; size: number }[]
  outline?: { title: string; sections: { title: string; sources: string[]; goal: string }[] } | null
  sections?: WriteupSection[]
  document?: string
  section_lines?: [number, number][]
  compile?: CompileResult | null
  outputs?: { tex?: string; pdf?: string }
  macros?: string[]
  theorems?: string[]
  output_name?: string
}

export type ChatListItem = {
  id: string
  kind: 'critic' | 'explore' | 'free'
  title: string
  updated_at: string
  created_at: string
  messages: number
  preview: string
}

export type ToolCall = { id?: string; function: { name: string; arguments: Record<string, unknown> | string } }

export type RouteMode = 'prove' | 'critic' | 'explore' | 'check' | 'literature' | 'referee' | 'writeup'

export type TranscriptItem = {
  role: 'user' | 'assistant' | 'tool' | 'review' | 'notice' | 'route'
  content: string
  time: string
  thinking?: string
  stats?: { eval_count?: number; prompt_eval_count?: number; done_reason?: string }
  tool_calls?: ToolCall[]
  tool_name?: string
  discarded?: boolean
  files?: string[]
  mode?: RouteMode | 'clarify'
  request?: string
  reason?: string
  question?: string
  status?: 'proposed' | 'starting' | 'queued' | 'started' | 'failed' | 'cancelled' | 'stopped' | 'dismissed'
  job_id?: string
  task?: string
  error?: string | null
  missing?: string[]
  effort?: string
}

export type Chat = {
  id: string
  mode: 'critic' | 'explore' | 'free'
  title: string
  settings: { think: boolean; model: string; online?: boolean }
  created_at: string
  updated_at: string
  transcript: TranscriptItem[]
  context_messages: number
}

export type StreamChunk = {
  file: string
  start: number
  end: number
  reset: boolean
  text: string
  thinking: string
  tool_calls: ToolCall[]
  done: boolean
  stats: Record<string, unknown>
}

export type Artifact = {
  name: string
  size: number
  text: string
  json?: unknown
  stream?: { text: string; thinking: string; tool_calls: ToolCall[]; done: boolean; stats: Record<string, unknown> }
}

export type WorkspaceFile = { path: string; size: number; kind: 'text' | 'pdf' }

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      method,
      credentials: 'same-origin',
      headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch {
    throw new ApiError(0, 'The harness is not reachable. Check that square-harness --gui is still running.')
  }
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    if (response.status === 401) authListeners.forEach((listener) => listener())
    throw new ApiError(response.status, (data && data.error) || response.statusText)
  }
  return data as T
}

const authListeners = new Set<() => void>()
export function onUnauthorized(listener: () => void) {
  authListeners.add(listener)
  return () => authListeners.delete(listener)
}

const get = <T,>(path: string) => request<T>('GET', path)
const post = <T,>(path: string, body: unknown = {}) => request<T>('POST', path, body)

export type StartResult = { task: TaskSummary; id: string | null }

export const api = {
  status: () => get<Status>('/api/status'),
  login: (token: string) => post<{ ok: boolean }>('/api/login', { token }),
  task: () => get<TaskSnapshot>('/api/task'),
  pause: () => post<{ task: TaskSummary }>('/api/task/pause'),
  decide: (id: string, approve: boolean) => post<{ ok: boolean }>(`/api/approvals/${id}`, { approve }),
  files: () => get<{ files: WorkspaceFile[] }>('/api/files'),

  proofs: () => get<{ jobs: ProofListItem[] }>('/api/proofs'),
  proof: (id: string) => get<ProofDetail>(`/api/proofs/${id}`),
  proofReport: (id: string) => get<{ report: string; ledger: string }>(`/api/proofs/${id}/report`),
  proofSources: (id: string) => get<{ sources: { path: string; sha256?: string; content: string }[] }>(`/api/proofs/${id}/sources`),
  proofArtifact: (id: string, name: string) => get<Artifact>(`/api/proofs/${id}/artifacts/${name}`),
  proofStream: (id: string, name: string, from = 0) => get<StreamChunk>(`/api/proofs/${id}/stream/${name}?from=${from}`),
  startProof: (body: Record<string, unknown>) => post<StartResult>('/api/proofs', body),
  resumeProof: (id: string, queue = false) => post<StartResult>(`/api/proofs/${id}/resume`, { queue }),

  researches: () => get<{ jobs: ResearchListItem[] }>('/api/research'),
  research: (id: string) => get<ResearchDetail>(`/api/research/${id}`),
  researchSources: (id: string) => get<{ sources: { id: string; path: string; sha256: string; content: string }[] }>(`/api/research/${id}/sources`),
  researchArtifact: (id: string, name: string) => get<Artifact>(`/api/research/${id}/artifacts/${name}`),
  researchStream: (id: string, name: string, from = 0) => get<StreamChunk>(`/api/research/${id}/stream/${name}?from=${from}`),
  startResearch: (body: Record<string, unknown>) => post<StartResult>('/api/research', body),
  resumeResearch: (id: string, queue = false) => post<StartResult>(`/api/research/${id}/resume`, { queue }),

  folders: (path = '') => get<FolderListing>('/api/folders?path=' + encodeURIComponent(path)),
  createFolder: (path: string, name: string) => post<{ path: string }>('/api/folders', { path, name }),
  openFolder: (path: string) => post<{ workspace: string }>('/api/workspace', { path }),
  saveSettings: (read: Partial<ReadSettings>) => post<{ read: ReadSettings }>('/api/settings', { read }),
  upload: async (file: File) => {
    let response: Response
    try {
      response = await fetch('/api/files/upload?name=' + encodeURIComponent(file.name), {
        method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/octet-stream' }, body: file,
      })
    } catch {
      throw new ApiError(0, 'The harness is not reachable.')
    }
    const data = await response.json().catch(() => ({}))
    if (!response.ok) throw new ApiError(response.status, data.error || response.statusText)
    return data as { path: string }
  },
  templates: () => get<{ templates: Template[] }>('/api/templates'),
  startWriteup: (body: Record<string, unknown>) => post<StartResult>('/api/writeup', body),
  researchPdfUrl: (id: string) => `/api/research/${id}/pdf`,
  cancelQueued: (task: string) => post<{ task: TaskSummary }>(`/api/queue/${task}/cancel`),
  route: (id: string, content: string, files: string[]) => post<{ ok: boolean }>(`/api/chats/${id}/route`, { content, files }),
  startRoute: (id: string, index: number, body: { mode: string; request: string; files: string[]; limits?: Record<string, number> }) =>
    post<{ task: TaskSummary }>(`/api/chats/${id}/routes/${index}/start`, body),
  cancelRoute: (id: string, index: number) => post<{ task: TaskSummary }>(`/api/chats/${id}/routes/${index}/cancel`),
  dismissRoute: (id: string, index: number) => post<{ item: TranscriptItem }>(`/api/chats/${id}/routes/${index}/dismiss`),
  chats: () => get<{ chats: ChatListItem[] }>('/api/chats'),
  chat: (id: string) => get<Chat>(`/api/chats/${id}`),
  createChat: (mode: string, think?: boolean, online?: boolean) => post<Chat>('/api/chats', { mode, think, online }),
  chatSettings: (id: string, settings: { think?: boolean; online?: boolean }) => post<Chat>(`/api/chats/${id}/settings`, settings),
  send: (id: string, content: string, files: string[] = []) => post<{ task: TaskSummary }>(`/api/chats/${id}/messages`, { content, files }),
  review: (id: string) => post<{ task: TaskSummary }>(`/api/chats/${id}/review`),
}
