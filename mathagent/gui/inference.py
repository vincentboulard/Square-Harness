"""Fair, cancelable admission to the GUI's shared inference connections.

Each task keeps its own token budget. This pool only bounds simultaneous model
streams, leaving token throughput to the inference server's scheduler.
"""
from __future__ import annotations

from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass
import math
import threading
import time
from typing import Hashable, Iterator


@dataclass(eq=False)
class _Waiter:
    owner: Hashable
    cancel: threading.Event
    deadline: float | None = None


class InferencePool:
    """Share a bounded number of streams between registered task owners.

    An owner may hold at most ceil(capacity / registered owners) streams. One
    task can therefore use both default slots; two tasks can each use one.
    Changes never interrupt streams already admitted. They take effect as those
    streams finish. The oldest eligible waiter enters next, so a busy owner's
    waiting workers cannot block another owner's request.
    """

    _CANCEL_POLL_SECONDS = 0.05

    def __init__(self, capacity: int = 2):
        self._validate_capacity(capacity)
        self._capacity = capacity
        self._condition = threading.Condition()
        self._owners: set[Hashable] = set()
        self._active: Counter[Hashable] = Counter()
        self._waiters: deque[_Waiter] = deque()

    @staticmethod
    def _validate_capacity(capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= 8:
            raise ValueError('Inference concurrency must be an integer between 1 and 8.')

    @property
    def capacity(self) -> int:
        with self._condition:
            return self._capacity

    def set_capacity(self, capacity: int) -> None:
        """Resize admission capacity, allowing existing streams to drain."""
        self._validate_capacity(capacity)
        with self._condition:
            self._capacity = capacity
            self._condition.notify_all()

    def register(self, owner: Hashable) -> None:
        """Register an active task. Repeated registration is harmless."""
        with self._condition:
            self._owners.add(owner)
            self._condition.notify_all()

    def unregister(self, owner: Hashable) -> None:
        """Retire a task and interrupt its waiters, preserving active streams."""
        with self._condition:
            self._owners.discard(owner)
            self._condition.notify_all()

    def _next_waiter(self) -> _Waiter | None:
        # Called with the condition held. Retired owners' active streams still
        # count against total capacity until their context managers exit.
        if sum(self._active.values()) >= self._capacity or not self._owners:
            return None
        allowance = (self._capacity + len(self._owners) - 1) // len(self._owners)
        now = time.monotonic()
        return next((waiter for waiter in self._waiters
                     if waiter.owner in self._owners and not waiter.cancel.is_set()
                     and (waiter.deadline is None or now < waiter.deadline)
                     and self._active[waiter.owner] < allowance), None)

    def _acquire(self, owner: Hashable, cancel: threading.Event, deadline: float | None) -> None:
        waiter = _Waiter(owner, cancel, deadline)
        with self._condition:
            if owner not in self._owners:
                raise ValueError('Register the inference owner before acquiring a slot.')
            self._waiters.append(waiter)
            self._condition.notify_all()
            try:
                while True:
                    if cancel.is_set() or owner not in self._owners:
                        raise KeyboardInterrupt
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError('time budget exhausted while waiting for inference')
                    if self._next_waiter() is waiter:
                        if deadline is not None and time.monotonic() >= deadline:
                            raise TimeoutError('time budget exhausted while waiting for inference')
                        self._waiters.remove(waiter)
                        self._active[owner] += 1
                        # Another eligible waiter may use remaining capacity.
                        self._condition.notify_all()
                        return
                    # A plain threading.Event does not notify this condition.
                    # Poll only while waiting, so pause remains responsive.
                    timeout = self._CANCEL_POLL_SECONDS
                    if deadline is not None:
                        timeout = min(timeout, max(0, deadline - time.monotonic()))
                    self._condition.wait(timeout)
            finally:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    self._condition.notify_all()

    @contextmanager
    def slot(self, owner: Hashable, cancel: threading.Event, *, deadline: float | None = None) -> Iterator[None]:
        """Hold a slot, respecting an optional monotonic admission deadline.

        Expiry while waiting raises TimeoutError without admitting the stream.
        Cancellation takes precedence when both conditions are already true.
        """
        if deadline is not None and (isinstance(deadline, bool)
                                    or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
            raise ValueError('The inference deadline must be a finite monotonic time.')
        self._acquire(owner, cancel, deadline)
        try:
            yield
        finally:
            with self._condition:
                self._active[owner] -= 1
                if not self._active[owner]:
                    del self._active[owner]
                self._condition.notify_all()
