"""Model transports with the harness's native message/event interface."""
import copy
import json
import time
from http.client import HTTPException
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler

from .agent import AgentError, Ollama, _NoModelRedirects


class TokenBudgetError(AgentError):
    """Fatal server cap violation, retaining measured usage for durable charging."""
    def __init__(self, stats, requested_cap):
        self.stats = dict(stats)
        self.requested_cap = requested_cap
        super().__init__('Model server reported completion_tokens above the requested output cap '
                         f'({self.stats["eval_count"]} > {requested_cap})')


def create_client(backend, host, timeout=600):
    if backend == 'ollama':
        client = Ollama(host, timeout=timeout)
        client.backend = 'ollama'
        return client
    if backend == 'openai':
        return OpenAICompatible(host, timeout=timeout)
    if backend == 'llamacpp':
        return LlamaCpp(host, timeout=timeout)
    raise AgentError(f'Unknown model backend: {backend}')


def _messages(messages):
    """Convert native tool history without modifying saved harness artifacts."""
    result, pending = [], []
    for index, original in enumerate(messages):
        role = original.get('role')
        if role not in {'system', 'user', 'assistant', 'tool'}:
            raise AgentError(f'Unsupported message role: {role}')
        if role != 'tool' and pending:
            raise AgentError('Tool history has missing tool results')
        message = {'role': role, 'content': original.get('content', '')}
        if role == 'assistant':
            reasoning = original.get('thinking', original.get('reasoning', original.get('reasoning_content')))
            if reasoning is not None:
                message['reasoning'] = reasoning
            calls = []
            for number, call in enumerate(original.get('tool_calls') or []):
                function = call.get('function', {})
                name = function.get('name')
                arguments = _arguments(function.get('arguments', {}))
                call_id = call.get('id') or f'call_{index}_{number}'
                if not isinstance(name, str) or not name or not isinstance(call_id, str):
                    raise AgentError('Malformed tool call in history')
                if any(item['id'] == call_id for item in pending):
                    raise AgentError('Duplicate tool call ID in history')
                calls.append({'id': call_id, 'type': 'function', 'function': {
                    'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)}})
                pending.append({'id': call_id, 'name': name})
            if calls:
                message['tool_calls'] = calls
        elif role == 'tool':
            call_id, name = original.get('tool_call_id'), original.get('tool_name', original.get('name'))
            match = next((item for item in pending if
                          (item['id'] == call_id if call_id else name is None or item['name'] == name)), None)
            if match is None:
                raise AgentError('Tool result has no matching assistant tool call')
            message['tool_call_id'] = match['id']
            pending.remove(match)
        result.append(message)
    if pending:
        raise AgentError('Tool history has missing tool results')
    return result


def _arguments(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError) as exc:
            raise AgentError('Malformed JSON tool arguments') from exc
    if not isinstance(value, dict):
        raise AgentError('Tool arguments must be a JSON object')
    return value


class OpenAICompatible:
    """Local vLLM /v1 adapter; no credentials, redirects or ambient proxies."""
    backend = 'openai'

    def __init__(self, host='http://localhost:8000', timeout=600):
        self.host = host.rstrip('/')
        parsed = urlsplit(self.host)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise AgentError('Model host must be an HTTP(S) URL without credentials, query or fragment')
        self.base_url = self.host if parsed.path.rstrip('/').endswith('/v1') else self.host + '/v1'
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}), _NoModelRedirects())

    def request(self, endpoint, data=None):
        return self._request_url(self.base_url + endpoint, data)

    def _request_url(self, url, data=None):
        request = Request(url,
                          data=None if data is None else json.dumps(data, ensure_ascii=False).encode(),
                          headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream' if data else 'application/json'})
        try:
            return self.opener.open(request, timeout=self.timeout)
        except HTTPError as exc:
            with exc:
                detail = exc.read(2000).decode(errors='replace')
            raise AgentError(f'Model server HTTP {exc.code}: {detail}') from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise AgentError(f'Cannot reach model server at {self.host}: {exc}') from exc

    def models(self):
        try:
            with self.request('/models') as response:
                data = json.load(response)
            models = data['data']
            if not isinstance(models, list) or any(not isinstance(item.get('id'), str) for item in models):
                raise ValueError('Invalid model list')
            return [item['id'] for item in models]
        except (ValueError, KeyError, TypeError, AttributeError, OSError, HTTPException) as exc:
            raise AgentError('Malformed model list from server') from exc

    def _payload(self, payload):
        options = payload.get('options', {})
        data = {'model': payload['model'], 'messages': _messages(payload['messages']),
                'stream': True, 'stream_options': {'include_usage': True},
                'chat_template_kwargs': {'enable_thinking': payload.get('think', True), 'preserve_thinking': True}}
        cap = options.get('num_predict')
        if cap is not None:
            if type(cap) is not int or cap <= 0:
                raise AgentError('OpenAI-compatible generation needs a positive output token cap')
            data['max_completion_tokens'] = cap
        for field in ('temperature', 'seed', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'frequency_penalty', 'stop'):
            if field in options:
                data[field] = options[field]
        if 'repeat_penalty' in options:
            data['repetition_penalty'] = options['repeat_penalty']
        if payload.get('reasoning_effort') is not None:
            data['reasoning_effort'] = payload['reasoning_effort']
        if payload.get('tools'):
            data['tools'] = copy.deepcopy(payload['tools'])
            data['tool_choice'] = 'auto'
        schema = payload.get('format')
        if schema == 'json':
            data['response_format'] = {'type': 'json_object'}
        elif isinstance(schema, dict):
            data['structured_outputs'] = {'json': copy.deepcopy(schema)}
        elif schema is not None:
            raise AgentError('Unsupported structured output format')
        return data

    @staticmethod
    def _sse(response, deadline=None):
        lines, size = [], 0
        for raw in response:
            # Check even comment/keepalive lines that do not yield model events.
            # Blocking reads still obey the socket's idle timeout; this is a
            # best-effort elapsed guard, not a hard process-level deadline.
            if deadline is not None and time.monotonic() >= deadline:
                raise AgentError('Model stream exceeded its elapsed-time guard')
            line = raw.decode('utf-8').rstrip('\r\n')
            if not line:
                if lines:
                    yield '\n'.join(lines)
                    lines, size = [], 0
            elif line.startswith('data:'):
                value = line[5:]
                lines.append(value[1:] if value.startswith(' ') else value)
                size += len(raw)
                if size > 4 * 1024 * 1024:
                    raise AgentError('Model server SSE event exceeds 4 MiB')
            # Comments and event/id/retry fields do not carry model output.
        if lines:
            yield '\n'.join(lines)

    def stream(self, payload):
        data = self._payload(payload)
        yield from self._stream(data)

    def _stream(self, data):
        calls, finish, usage = {}, None, None
        requested_cap = data.get('max_completion_tokens', data.get('max_tokens', float('inf')))
        deadline = time.monotonic() + self.timeout
        try:
            with self.request('/chat/completions', data) as response:
                for raw in self._sse(response, deadline):
                    if raw.strip() == '[DONE]':
                        if finish is None or usage is None:
                            raise AgentError('Model stream ended without finish reason and token usage')
                        complete_calls, ids = [], set()
                        # A capped tool request can end inside JSON arguments.
                        # Preserve the length result/usage for controller recovery,
                        # but never expose unfinished calls to a tool executor.
                        for index in sorted(calls) if finish != 'length' else []:
                            call = calls[index]
                            if not call['id'] or not call['function']['name'] or call['id'] in ids:
                                raise AgentError('Malformed or duplicate streamed tool call')
                            ids.add(call['id'])
                            call['function']['arguments'] = _arguments(call['function']['arguments'])
                            complete_calls.append(call)
                        if finish == 'tool_calls' and not complete_calls:
                            raise AgentError('Model finished with tool_calls but supplied no calls')
                        event = {'message': {'tool_calls': complete_calls} if complete_calls else {},
                                 'done': True, 'done_reason': finish,
                                 'eval_count': usage['completion_tokens'], 'prompt_eval_count': usage['prompt_tokens']}
                        yield event
                        return
                    chunk = json.loads(raw)
                    if not isinstance(chunk, dict):
                        raise AgentError('Malformed SSE model event')
                    if 'error' in chunk:
                        raise AgentError(f'Model stream error: {chunk["error"]}')
                    if chunk.get('usage') is not None:
                        current = chunk['usage']
                        if not isinstance(current, dict) or any(type(current.get(key)) is not int or current[key] < 0 for key in ('prompt_tokens', 'completion_tokens')):
                            raise AgentError('Missing or invalid token usage in model stream')
                        if current['completion_tokens'] > requested_cap:
                            raise TokenBudgetError({'eval_count': current['completion_tokens'],
                                                    'prompt_eval_count': current['prompt_tokens']},
                                                   requested_cap)
                        usage = current
                    choices = chunk.get('choices', [])
                    if not isinstance(choices, list) or len(choices) > 1:
                        raise AgentError('Expected one streamed completion choice')
                    for choice in choices:
                        if not isinstance(choice, dict) or choice.get('index', 0) != 0:
                            raise AgentError('Unexpected streamed completion index')
                        delta = choice.get('delta') or {}
                        if not isinstance(delta, dict):
                            raise AgentError('Malformed streamed completion delta')
                        if finish is not None and delta:
                            raise AgentError('Model output received after finish reason')
                        message = {}
                        for target, value in [('content', delta.get('content')),
                                              ('thinking', delta.get('reasoning') if delta.get('reasoning') is not None else delta.get('reasoning_content'))]:
                            if value is not None and not isinstance(value, str):
                                raise AgentError('Non-text content in model stream')
                            if value:
                                message[target] = value
                        for part in delta.get('tool_calls') or []:
                            index = part.get('index')
                            if type(index) is not int or not 0 <= index < 16:
                                raise AgentError('Invalid streamed tool call index')
                            call = calls.setdefault(index, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                            if part.get('type', 'function') != 'function':
                                raise AgentError('Unsupported streamed tool type')
                            function = part.get('function') or {}
                            for container, key, value in [(call, 'id', part.get('id')),
                                                          (call['function'], 'name', function.get('name')),
                                                          (call['function'], 'arguments', function.get('arguments'))]:
                                if value is not None:
                                    if not isinstance(value, str):
                                        raise AgentError('Malformed streamed tool call fragment')
                                    container[key] += value
                        if message:
                            yield {'message': message, 'done': False}
                        reason = choice.get('finish_reason')
                        if reason is not None:
                            if finish is not None or reason not in {'stop', 'length', 'tool_calls'}:
                                raise AgentError(f'Unsupported or repeated model finish reason: {reason}')
                            finish = reason
            raise AgentError('Model stream ended before [DONE]; partial output is incomplete')
        except (ValueError, UnicodeError, TypeError, KeyError, AttributeError, OSError, HTTPException) as exc:
            raise AgentError(f'Malformed or interrupted model stream: {exc}') from exc


class LlamaCpp(OpenAICompatible):
    """llama-server chat API with exact, template-aware per-slot context checks.

    The supported server contract is pinned in the deployment files. Its
    /chat/completions/input_tokens route applies the same template, tool/schema
    handling and special-token policy as generation, without running inference.
    """
    backend = 'llamacpp'

    def __init__(self, host='http://localhost:8000', timeout=600):
        super().__init__(host, timeout)
        self.root_url = self.base_url[:-len('/v1')]
        self._properties = None

    def _payload(self, payload):
        if payload.get('reasoning_effort') is not None:
            raise AgentError('The llama.cpp preset uses think on/off, not reasoning_effort levels')
        if type(payload.get('think', True)) is not bool:
            raise AgentError('llama.cpp thinking must be a boolean')
        data = super()._payload(payload)
        if 'max_completion_tokens' not in data:
            raise AgentError('llama.cpp requires a positive output token cap')
        data['max_tokens'] = data.pop('max_completion_tokens')
        data['reasoning_format'] = 'deepseek'
        data['chat_template_kwargs'] = {'enable_thinking': payload.get('think', True),
                                        'preserve_reasoning': True}
        for message in data['messages']:
            if 'reasoning' in message:
                message['reasoning_content'] = message.pop('reasoning')
        if 'repetition_penalty' in data:
            data['repeat_penalty'] = data.pop('repetition_penalty')
        if 'structured_outputs' in data:
            schema = data.pop('structured_outputs')['json']
            data['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'square_harness_response', 'strict': True, 'schema': schema}}
        return data

    def _json(self, endpoint, data=None, *, root=False):
        try:
            request = self._request_url(self.root_url + endpoint, data) if root else self.request(endpoint, data)
            with request as response:
                result = json.load(response)
            if not isinstance(result, dict):
                raise ValueError('Expected a JSON object')
            return result
        except (ValueError, KeyError, TypeError, AttributeError, OSError, HTTPException) as exc:
            raise AgentError(f'Malformed llama.cpp response from {endpoint}: {exc}') from exc

    def properties(self, *, refresh=False):
        if refresh or self._properties is None:
            props = self._json('/props', root=True)
            settings = props.get('default_generation_settings')
            context = settings.get('n_ctx') if isinstance(settings, dict) else None
            if type(context) is not int or context <= 0:
                raise AgentError('llama.cpp /props must report a positive per-slot n_ctx')
            if not isinstance(props.get('chat_template'), str) or not props['chat_template'].strip():
                raise AgentError('llama.cpp must expose its configured chat template through /props')
            self._properties = props
        return copy.deepcopy(self._properties)

    def _count_input_tokens(self, data):
        count = self._json('/chat/completions/input_tokens', data).get('input_tokens')
        if type(count) is not int or count < 0:
            raise AgentError('llama.cpp must return an integer input_tokens count')
        return count

    def count_input_tokens(self, payload):
        """Count a native request after the server applies its exact chat template."""
        return self._count_input_tokens(self._payload(payload))

    def stream(self, payload):
        data = self._payload(payload)
        context = payload.get('options', {}).get('num_ctx')
        if type(context) is not int or context <= 0:
            raise AgentError('llama.cpp requires an explicit positive num_ctx context budget')
        timeout, started = self.timeout, time.monotonic()

        def remaining_timeout():
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise AgentError('llama.cpp request exceeded its elapsed-time guard before generation')
            self.timeout = remaining

        try:
            remaining_timeout()
            slot_context = self.properties()['default_generation_settings']['n_ctx']
            if context > slot_context:
                raise AgentError(f'Requested context {context} exceeds llama.cpp per-slot context {slot_context}; '
                                 'restart the server with the required slot capacity or explicitly lower --ctx')
            remaining_timeout()
            prompt_tokens = self._count_input_tokens(data)
            # llama-server stops at prompt.n_tokens()+1 >= slot.n_ctx. Reserve
            # that boundary token as well as every requested output token.
            if prompt_tokens + data['max_tokens'] + 1 > context:
                raise AgentError(f'Context budget exceeded: {prompt_tokens} prompt tokens + '
                                 f'{data["max_tokens"]} output tokens + 1 boundary token > {context}; '
                                 'the statement and output allowance were not truncated')
            remaining_timeout()
            for event in self._stream(data):
                if event.get('done') and event.get('prompt_eval_count') != prompt_tokens:
                    raise AgentError('llama.cpp prompt token count changed between validation and generation; '
                                     'refusing a potentially truncated or differently formatted completion')
                yield event
        finally:
            self.timeout = timeout
