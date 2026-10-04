"""Durable inference accounting shared by a bounded concurrent Assistant tree."""
import json
import math
import threading
import time

from .agent import AgentError
from .proof import count_input


class AssistantBudget(AgentError):
    pass


class _Pool:
    def __init__(self, state, checkpoint, concurrency):
        self.state = state
        self.lock = threading.RLock()
        self.slots = threading.BoundedSemaphore(concurrency)
        self.checkpoint = checkpoint
        self.started = time.monotonic()
        self.seconds_before = state['seconds']


class BudgetClient:
    def __init__(self, client, *, state=None, max_tokens=60000, max_input_tokens=240000,
                 max_seconds=900, checkpoint=lambda: None, concurrency=2, pool=None,
                 allowance=None, usage=None):
        self.client = client
        budget = dict(state or {'tokens': 0, 'input_tokens': 0, 'seconds': 0,
                               'max_tokens': max_tokens, 'max_input_tokens': max_input_tokens,
                               'max_seconds': max_seconds})
        for key in ('tokens', 'input_tokens', 'seconds', 'max_tokens', 'max_input_tokens', 'max_seconds'):
            value = budget.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid Assistant budget: ' + key)
        self.pool = pool or _Pool(budget, checkpoint, concurrency)
        self.allowance = allowance
        self.usage = usage if usage is not None else {'tokens': 0, 'input_tokens': 0}

    def __getattr__(self, name):
        return getattr(self.client, name)

    @property
    def timeout(self):
        return self.client.timeout

    @timeout.setter
    def timeout(self, value):
        self.client.timeout = value

    def fork(self, client, *, allowance, usage):
        return BudgetClient(client, pool=self.pool, allowance=allowance, usage=usage)

    @property
    def remaining_tokens(self):
        with self.pool.lock:
            remaining = max(0, self.pool.state['max_tokens'] - self.pool.state['tokens'])
            if self.allowance is not None:
                remaining = min(remaining, max(0, self.allowance - self.usage['tokens']))
            return remaining

    @property
    def remaining_seconds(self):
        return max(0, self.pool.state['max_seconds'] - self.snapshot()['seconds'])

    def snapshot(self):
        with self.pool.lock:
            return {**self.pool.state, 'seconds': self.pool.seconds_before + time.monotonic() - self.pool.started}

    def stream(self, payload):
        cancel = getattr(self.client, 'cancel', None)
        if cancel is not None and cancel.is_set():
            raise KeyboardInterrupt
        while not self.pool.slots.acquire(timeout=0.1):
            if getattr(self.client, 'cancel', None) is not None and self.client.cancel.is_set():
                raise KeyboardInterrupt
            if self.remaining_seconds <= 0:
                raise AssistantBudget('Assistant time budget exhausted while waiting for inference.')
        try:
            if cancel is not None and cancel.is_set():
                raise KeyboardInterrupt
            yield from self._stream(payload)
        finally:
            self.pool.slots.release()

    def _stream(self, payload):
        cap = payload.get('options', {}).get('num_predict')
        if type(cap) is not int or cap <= 0:
            raise AssistantBudget('Model generation needs a positive output cap.')
        if self.remaining_seconds <= 0:
            raise AssistantBudget('Assistant time budget exhausted.')
        count, method = count_input(self.client, payload)
        if method != 'server token count':
            count += len(json.dumps(payload.get('tools', []), ensure_ascii=False).encode())
        if count + cap + 1 > payload['options']['num_ctx']:
            raise AgentError('The complete Assistant context does not fit. No hypotheses or worker results were clipped.')
        with self.pool.lock:
            if type(cap) is not int or cap <= 0 or cap > self.remaining_tokens:
                raise AssistantBudget('Assistant output budget cannot cover the next model call.')
            if self.pool.state['input_tokens'] + count > self.pool.state['max_input_tokens']:
                raise AssistantBudget('Assistant input budget exhausted; no mathematical context was clipped.')
            # Reserve before dispatch. Interrupted or unaccounted calls retain
            # their allowance, so a resume cannot spend those tokens twice.
            self.pool.state['tokens'] += cap
            self.pool.state['input_tokens'] += count
            self.usage['tokens'] += cap
            self.usage['input_tokens'] += count
        self.pool.checkpoint()
        previous_timeout = self.client.timeout
        self.client.timeout = max(0.1, min(previous_timeout, self.remaining_seconds))
        events = self.client.stream(payload)
        charged = False
        try:
            for event in events:
                if self.remaining_seconds <= 0:
                    raise AssistantBudget('Assistant time budget exhausted during generation.')
                if event.get('done'):
                    if charged:
                        raise AssistantBudget('Model stream contained duplicate completion events.')
                    charged = True
                    used = event.get('eval_count')
                    if used is not None:
                        if type(used) is not int or used < 0 or used > cap:
                            raise AssistantBudget('Model usage exceeded the reserved Assistant allowance.')
                        with self.pool.lock:
                            self.pool.state['tokens'] -= cap - used
                            self.usage['tokens'] -= cap - used
                    self.pool.checkpoint()
                yield event
        finally:
            if hasattr(events, 'close'):
                events.close()
            self.client.timeout = previous_timeout
            self.pool.checkpoint()
