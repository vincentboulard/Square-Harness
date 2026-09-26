"""Native Ollama HTTP client and bounded, backend-independent agent loop."""
import copy
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

from .prompts import SYSTEM, MODES


class AgentError(Exception):
    pass


class _NoModelRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a manuscript or local-only request to a redirect target.
        # Configure the final Ollama host explicitly instead.
        return None


class Ollama:
    backend = 'ollama'

    def __init__(self, host='http://localhost:11434', timeout=600):
        self.host = host.rstrip('/')
        self.timeout = timeout
        # Local model traffic should not inherit an HTTP proxy configuration.
        self.opener = build_opener(ProxyHandler({}), _NoModelRedirects())

    def request(self, endpoint, data=None):
        req = Request(self.host + endpoint,
                      data=None if data is None else json.dumps(data).encode(),
                      headers={'Content-Type': 'application/json'})
        try:
            return self.opener.open(req, timeout=self.timeout)
        except HTTPError as e:
            with e:
                detail = e.read(2000).decode(errors='replace')
            raise AgentError(f'Ollama HTTP {e.code}: {detail}') from e
        except (URLError, TimeoutError, OSError) as e:
            raise AgentError(f'Cannot reach Ollama at {self.host}: {e}. Check ollama serve.') from e

    def models(self):
        with self.request('/api/tags') as response:
            return [m['name'] for m in json.load(response).get('models', [])]

    def stream(self, payload):
        with self.request('/api/chat', payload) as response:
            for line in response:
                if not line.strip():
                    continue
                event = json.loads(line)
                if 'error' in event:
                    raise AgentError(str(event['error']))
                yield event


class Agent:
    def __init__(self, client, workspace, model='qwen3.8:27b', ctx=8192,
                 predict=4096, think=True, mode='prove', max_rounds=8,
                 seed=None, temperature=0.6, top_p=0.95):
        self.client, self.workspace = client, workspace
        self.model, self.ctx, self.predict = model, ctx, predict
        self.think, self.mode, self.max_rounds = think, mode, max_rounds
        self.seed, self.temperature, self.top_p = seed, temperature, top_p
        self.history = []
        self.last_stats = {}

    def run(self, query, emit=lambda kind, value: None, fresh=False):
        # Only commit history when a full turn completes; interruptions cannot
        # leave a dangling assistant tool call in the next API request.
        messages = copy.deepcopy([] if fresh else self.history)
        messages.append({'role': 'user', 'content': query})
        system = {'role': 'system', 'content': SYSTEM + '\nMODE: ' + MODES[self.mode]}
        schemas = self.workspace.schemas()
        for round_no in range(self.max_rounds + 1):
            tools = schemas if round_no < self.max_rounds else []
            # Conservative byte budget, not an exact tokenizer. Drop ONLY whole
            # old user turns; never truncate a tool call/result pair or current turn.
            budget = max(2000, (self.ctx - self.predict - 512) * 3)
            def input_size():
                return len(json.dumps([system] + messages, ensure_ascii=False).encode()) + len(json.dumps(tools))
            while input_size() > budget:
                user_indices = [i for i, m in enumerate(messages) if m['role'] == 'user']
                if len(user_indices) < 2:
                    raise AgentError('Current turn exceeds the approximate input budget. Use smaller excerpts, '
                                     '/clear, increase /context, or reduce --predict. This turn was not saved.')
                messages = messages[user_indices[1]:]
                emit('notice', 'Dropped the oldest complete turn to fit the approximate context budget.')
            payload = {'model': self.model, 'messages': [system] + messages, 'stream': True,
                       'think': self.think, 'options': {'num_ctx': self.ctx, 'num_predict': self.predict,
                           'temperature': self.temperature, 'top_p': self.top_p}}
            if self.seed is not None:
                payload['options']['seed'] = (self.seed + round_no) % (2 ** 31)
            if tools:
                payload['tools'] = tools
            emit('start', '')
            assistant = {'role': 'assistant', 'content': ''}
            calls, thinking, done = [], '', False
            for event in self.client.stream(payload):
                msg = event.get('message', {})
                if msg.get('thinking'):
                    thinking += msg['thinking']
                    emit('thinking', msg['thinking'])
                if msg.get('content'):
                    assistant['content'] += msg['content']
                    emit('text', msg['content'])
                calls.extend(msg.get('tool_calls') or [])
                if event.get('done'):
                    self.last_stats = {k: event[k] for k in ('eval_count', 'eval_duration',
                        'prompt_eval_count', 'done_reason') if k in event}
                    done = True
            emit('end', '')
            if not done:
                raise AgentError('Model stream ended before its completion event. This turn was not saved.')
            if thinking:
                assistant['thinking'] = thinking
            if calls:
                assistant['tool_calls'] = calls
            messages.append(assistant)
            if self.last_stats.get('done_reason') == 'length':
                emit('notice', 'Generation hit its output limit; any displayed argument may be incomplete.')
            if calls:
                if not tools or len(calls) > 16:
                    raise AgentError('Tool budget exceeded; this turn was not saved. Previously approved actions remain applied.')
                for call in calls:
                    fn = call.get('function', {})
                    name = fn.get('name', '')
                    args = fn.get('arguments', {})
                    emit('tool', name + ' ' + json.dumps(args, ensure_ascii=False)[:240])
                    result = self.workspace.execute(name, args)
                    tool_message = {'role': 'tool', 'tool_name': name, 'content': result}
                    if call.get('id'):
                        tool_message['tool_call_id'] = call['id']
                    messages.append(tool_message)
                    emit('result', result[:200])
                if round_no + 1 == self.max_rounds:
                    emit('notice', 'Tool-round limit reached. Requesting a final answer without further tools.')
                continue
            if not assistant['content'].strip():
                emit('notice', 'No final answer received. Try --predict 6144 --ctx 16384 or /think off.')
            # Scratch thinking is retained within the tool loop but not persisted
            # between user turns: keep local context and exports manageable.
            for message in messages:
                message.pop('thinking', None)
            self.history = messages
            return assistant['content']
        raise AgentError('Agent round limit reached')
