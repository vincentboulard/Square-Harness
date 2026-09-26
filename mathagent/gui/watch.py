"""Publish saved-job changes and live model output by watching .mathagent/.

The engine saves state atomically and fsyncs every streamed event before it is
used, so these files are a dependable progress source for jobs started in the
browser or in a terminal alike.
"""
import json
from pathlib import Path
import threading

from ..ledger import _ARTIFACT, _ID, _read
from .store import _RESEARCH_ARTIFACT, stream_chunk


class Watcher(threading.Thread):
    def __init__(self, root, bus, interval=0.4):
        super().__init__(name='square-watcher', daemon=True)
        self.root = Path(root)
        self.bus = bus
        self.interval = interval
        self._halt = threading.Event()
        self._seen = {}
        self._live = {}
        self._offsets = {}

    def stop(self):
        self._halt.set()

    def run(self):
        self.scan(publish=False)
        while not self._halt.wait(self.interval):
            self.scan()

    def scan(self, publish=True):
        for kind, folder in (('proof', 'proofs'), ('research', 'research')):
            base = self.root / '.mathagent' / folder
            if not base.is_dir() or base.is_symlink():
                continue
            try:
                entries = list(base.iterdir())
            except OSError:
                continue
            for directory in entries:
                if _ID.fullmatch(directory.name) and not directory.is_symlink():
                    try:
                        self._check(kind, directory, publish)
                    except (OSError, ValueError):
                        continue  # one damaged job must not stop updates for the others

    def _check(self, kind, directory, publish):
        key = (kind, directory.name)
        info = (directory / 'state.json').lstat()
        stamp = (info.st_mtime_ns, info.st_size)
        if self._seen.get(key) != stamp:
            state = json.loads(_read(directory / 'state.json'))
            if not isinstance(state, dict):
                return
            self._seen[key] = stamp
            calls = state.get('calls') if isinstance(state.get('calls'), list) else []
            call = calls[-1] if calls and isinstance(calls[-1], dict) and calls[-1].get('status') == 'running' else None
            pattern = _ARTIFACT if kind == 'proof' else _RESEARCH_ARTIFACT
            live = (call['stream'], call.get('role', '')) if call and isinstance(call.get('stream'), str) and pattern.fullmatch(call['stream']) else None
            previous = self._live.get(key)
            if previous and previous != live:
                self._tail(kind, directory, previous, publish)  # deliver the stream's final lines
                self._offsets.pop((kind, directory.name, previous[0]), None)
            self._live[key] = live
            if publish:
                self.bus.publish('job', job=kind, id=directory.name, status=state.get('status'),
                                 phase=state.get('phase') or (state.get('pending') or {}).get('phase'),
                                 updated_at=state.get('updated_at'))
        if self._live.get(key):
            self._tail(kind, directory, self._live[key], publish)

    def _tail(self, kind, directory, live, publish):
        name, role = live
        offset_key = (kind, directory.name, name)
        path = directory / 'artifacts' / name
        if offset_key not in self._offsets:
            # Clients load earlier output through the API; only publish growth.
            self._offsets[offset_key] = stream_chunk(path, 0)['end'] if not publish else 0
        chunk = stream_chunk(path, self._offsets[offset_key])
        if chunk['end'] == chunk['start']:
            return
        self._offsets[offset_key] = chunk['end']
        if publish:
            self.bus.publish('stream', job=kind, id=directory.name, file=name, role=role,
                             start=chunk['start'], end=chunk['end'], text=chunk['text'],
                             thinking=chunk['thinking'], tool_calls=chunk['tool_calls'], done=chunk['done'])
