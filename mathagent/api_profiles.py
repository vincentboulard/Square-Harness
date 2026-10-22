"""Opt-in MLX chat transport; the existing vLLM adapter is unchanged.

The tokenizer and context_limit extensions below belong to Local LLM Lab.
Other MLX-compatible servers may omit tokenization; counting then falls back to
the harness's existing conservative estimate. JSON is prompted and validated by
the harness, never advertised as grammar-constrained decoding.
"""
import json
import os
from http.client import HTTPException

from .agent import AgentError
from .backends import OpenAICompatible


def resolve_api_profile(profile=None):
    selected = os.environ.get('SQUARE_OPENAI_PROFILE', 'default') if profile is None else profile
    if selected not in {'default', 'mlx'}:
        raise AgentError('API profile must be default or mlx')
    return selected


class MLX(OpenAICompatible):
    """MLX-compatible streaming, with optional Local LLM Lab token counting."""
    profile = 'mlx'

    def __init__(self, host='http://localhost:8000', timeout=600):
        super().__init__(host, timeout)
        if self.thinking_budget_policy:
            raise AgentError('The MLX profile does not support vLLM thinking-budget fractions')

    def _payload(self, payload):
        if payload.get('options', {}).get('seed') is not None:
            raise AgentError('MLX continuous batching does not support explicit seeds; omit --seed')
        if payload.get('reasoning_effort') is not None:
            raise AgentError('The MLX profile supports a thinking toggle, not reasoning_effort levels')
        data = super()._payload(payload)
        data['chat_template_kwargs'].pop('preserve_thinking', None)
        context = payload.get('options', {}).get('num_ctx')
        if context is not None:
            data['context_limit'] = context
        data.pop('response_format', None)
        data.pop('structured_outputs', None)
        schema = payload.get('format')
        if schema is not None:
            instruction = 'Return only a JSON object in the final answer, without Markdown fences.'
            if isinstance(schema, dict):
                instruction += ' It must match this JSON schema: ' + json.dumps(schema, ensure_ascii=False)
            data['messages'].append({'role': 'user', 'content': instruction})
        return data

    def count_input_tokens(self, payload):
        """Optional Local LLM Lab extension; errors never masquerade as counts."""
        if self._tokenize_supported is False:
            return None
        data = self._payload(payload)
        request = {key: data[key] for key in ('model', 'messages', 'tools') if key in data}
        request['thinking'] = bool(payload.get('think', True))
        response = self._request_url(self.base_url[:-3] + '/api/tokenize', request, optional=True)
        if response is None:
            self._tokenize_supported = False
            return None
        self._tokenize_supported = True
        try:
            with response:
                result = json.load(response)
            count, context = result['prompt_tokens'], result['max_context_tokens']
            if type(count) is not int or count < 0 or type(context) is not int or context <= 0:
                raise ValueError('Invalid count or model context')
        except (ValueError, KeyError, TypeError, OSError, HTTPException) as exc:
            raise AgentError('Malformed tokenizer response from model server') from exc
        self.observed_context_tokens = context
        requested = payload.get('options', {}).get('num_ctx')
        if type(requested) is int and requested > context:
            raise AgentError(f'Requested context {requested} exceeds server context {context}')
        return count
