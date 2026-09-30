"""Read-only views of saved proof and research jobs, and the interface's chats.

Job views never write engine state. Chat sessions are the interface's own
records: the terminal keeps chat in memory, while a browser tab or phone needs
the conversation to survive a reload.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
import uuid

from ..ledger import ProofStore, _ID, _ARTIFACT, _atomic_write, _directory, _read, _regular, _now
from ..research import _job_directory, _read_state as _read_research
from ..tools import READ_GROUPS, SKIP, TEXT
from ..writeup import TEMPLATE_SUFFIXES

CHAT_MODES = ('critic', 'explore', 'free')
# How the interface names a suggestion in chat previews: critic and explore answers
# are both answers in the conversation, and the referee workflow is called Review.
ROUTE_NAMES = {'prove': 'Proof', 'critic': 'Answer', 'explore': 'Answer', 'literature': 'Literature report',
               'referee': 'Review', 'writeup': 'Write-up', 'clarify': 'Question'}
UPLOAD_SUFFIXES = {'.tex', '.sty', '.cls', '.bib', '.md', '.txt', '.pdf', '.py'}
UPLOAD_BYTES = 20 * 1024 * 1024
_chat_lock = threading.RLock()
CHAT_BYTES = 16 * 1024 * 1024
STREAM_BYTES = 4 * 1024 * 1024
_RESEARCH_ARTIFACT = re.compile(r'[0-9]{4,}-[A-Za-z0-9_-]+\.md\Z')
_cache_lock = threading.Lock()
_cache = {}


class NotFound(LookupError):
    """A requested job, chat, artifact or approval does not exist."""


def one_line(text, limit=80):
    text = ' '.join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + '…'


def _cached(path, load):
    """Parse a state file once per (mtime, size); listings run on every job event."""
    info = path.lstat()
    key = (str(path), info.st_mtime_ns, info.st_size)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    value = load()
    with _cache_lock:
        if len(_cache) > 512:
            _cache.clear()
        _cache[key] = value
    return value


def locked(directory):
    """True while an engine process holds the job's run lock.

    Probing takes a shared lock for microseconds and is only used for jobs whose
    saved status says running, to tell live work from an interrupted process.
    """
    try:
        fd = os.open(directory / 'run.lock', os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def parse_stream(text):
    """Accumulate an Ollama NDJSON stream; a killed writer may leave a partial last line."""
    out = {'text': '', 'thinking': '', 'tool_calls': [], 'done': False, 'stats': {}}
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            break
        if not isinstance(event, dict):
            continue
        message = event.get('message') if isinstance(event.get('message'), dict) else {}
        for key, field in (('text', 'content'), ('thinking', 'thinking')):
            if isinstance(message.get(field), str):
                out[key] += message[field]
        if isinstance(message.get('tool_calls'), list):
            out['tool_calls'].extend(message['tool_calls'])
        if event.get('done'):
            out['done'] = True
            out['stats'] = {k: event[k] for k in ('eval_count', 'prompt_eval_count', 'done_reason', 'eval_duration')
                            if k in event}
    return out


def stream_chunk(path, start=0):
    """Complete lines of a stream file from byte offset `start`."""
    if type(start) is not int or start < 0:
        raise ValueError('Stream offset must be a nonnegative integer')
    _regular(path)
    size = path.lstat().st_size
    reset = start > size
    if reset:
        start = 0
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        stream.seek(start)
        data = stream.read(min(size - start, STREAM_BYTES))
    cut = data.rfind(b'\n') + 1
    parsed = parse_stream(data[:cut].decode('utf-8', errors='replace'))
    return {'file': path.name, 'start': start, 'end': start + cut, 'reset': reset, **parsed}


def _json_artifact(name, text):
    out = {'name': name, 'size': len(text.encode()), 'text': text}
    if name.endswith('.jsonl') or '-stream' in name:
        out['stream'] = parse_stream(text)
    else:
        try:
            out['json'] = json.loads(text)
        except ValueError:
            pass
    return out


def _artifacts(directory, pattern):
    found = []
    for path in sorted((directory / 'artifacts').iterdir()):
        if pattern.fullmatch(path.name) and not path.is_symlink():
            found.append({'name': path.name, 'size': path.lstat().st_size})
    return found


# -- proofs -----------------------------------------------------------------

def _proof_directory(root, proof_id):
    if not isinstance(proof_id, str) or not _ID.fullmatch(proof_id):
        raise ValueError('Invalid proof ID')
    directory = Path(root) / '.mathagent' / 'proofs' / proof_id
    if not directory.exists():
        raise NotFound('Unknown proof job')
    for path in (directory.parent.parent, directory.parent, directory, directory / 'artifacts'):
        _directory(path)
    return directory


def _budget(used, limit):
    return {'used': used, 'limit': limit}


def _review_counts(state):
    """Candidates by review status. Legacy v1 jobs recorded claims instead and show none."""
    counts = {}
    for candidate in state['candidates'] if state['version'] == 2 else []:
        status = candidate.get('review_status', 'unreviewed')
        counts[status] = counts.get(status, 0) + 1
    return counts


def _candidate_text(store, candidate):
    """A candidate's saved text, or None when the file no longer matches its digest."""
    try:
        text = store.read_artifact(candidate['artifact'])
    except ValueError:
        return None
    return text if hashlib.sha256(text.encode()).hexdigest() == candidate.get('sha256') else None


def list_proofs(root, active=None):
    root = Path(root)
    base = root / '.mathagent' / 'proofs'
    if not base.is_dir() or base.is_symlink():
        return []
    jobs = []
    for path in base.iterdir():
        if not _ID.fullmatch(path.name) or path.is_symlink():
            continue
        try:
            state = _cached(path / 'state.json', lambda: ProofStore.load(root, path.name).state)
        except (OSError, ValueError):
            continue
        running = state['status'] == 'running' and (path.name == active or locked(path))
        jobs.append({'id': state['id'], 'kind': 'proof', 'version': state['version'], 'status': state['status'],
                     'title': one_line(state['goal'], 140), 'updated_at': state['updated_at'],
                     'created_at': state['created_at'], 'running': running,
                     'sources': [s['path'] for s in state['sources']],
                     'rounds': _budget(state['rounds_started'], state['settings']['max_rounds']),
                     'tokens': _budget(state['tokens_charged'], state['settings']['max_tokens']),
                     'reviews': _review_counts(state)})
    return sorted(jobs, key=lambda job: job['updated_at'], reverse=True)


def proof_detail(root, proof_id, active=None):
    directory = _proof_directory(root, proof_id)
    store = ProofStore.load(root, proof_id)
    s = store.state
    running = s['status'] == 'running' and (proof_id == active or locked(directory))
    call = next((c for c in reversed(s['calls']) if c.get('status') == 'running'), None) if running else None
    settings = s['settings']
    detail = {
        'id': s['id'], 'kind': 'proof', 'version': s['version'], 'legacy': s['version'] != 2,
        'goal': s['goal'], 'status': s['status'], 'stop_reason': s.get('stop_reason'),
        'recovery_notice': s.get('recovery_notice'),
        'created_at': s['created_at'], 'updated_at': s['updated_at'], 'revision': s['revision'],
        'settings': settings,
        'budget': {'rounds': _budget(s['rounds_started'], settings['max_rounds']),
                   'tokens': _budget(s['tokens_charged'], settings['max_tokens']),
                   'seconds': _budget(round(s['seconds_used'], 1), settings['max_seconds'])},
        'sources': [{'path': src['path'], 'sha256': src.get('sha256'),
                     'lines': len(src['content'].splitlines())} for src in s['sources']],
        'calls': s['calls'], 'running': running,
        'live': {'file': call['stream'], 'role': call['role'], 'round': call['round']} if call else None,
        'artifacts': _artifacts(directory, _ARTIFACT),
    }
    if s['version'] != 2:
        # A v0.4 job: readable, never resumed. Its proof.md is the audited candidate, if any.
        proof = directory / 'proof.md'
        detail.update(candidates=[], reviews=[], pending=None, selected_candidate=None, initial_candidate=None,
                      selection_history=[], answer=_read(proof) if proof.exists() else None)
        return detail
    selected = next((c for c in s['candidates'] if c['id'] == s['selected_candidate']), None)
    detail.update(candidates=s['candidates'], reviews=s['reviews'], pending=s['pending'],
                  selected_candidate=s['selected_candidate'], initial_candidate=s['initial_candidate'],
                  selection_history=s.get('selection_history', []),
                  answer=_candidate_text(store, selected) if selected else None)
    return detail


def proof_sources(root, proof_id):
    _proof_directory(root, proof_id)
    return [{'path': src['path'], 'sha256': src.get('sha256'), 'content': src['content']}
            for src in ProofStore.load(root, proof_id).state['sources']]


def proof_report(root, proof_id):
    directory = _proof_directory(root, proof_id)
    out = {}
    for name in ('report', 'ledger'):
        path = directory / (name + '.md')
        out[name] = _read(path) if path.exists() else ''
    return out


def proof_artifact(root, proof_id, name):
    directory = _proof_directory(root, proof_id)
    if not _ARTIFACT.fullmatch(name):
        raise ValueError('Invalid artifact name')
    path = directory / 'artifacts' / name
    if not path.exists():
        raise NotFound('Unknown artifact')
    return _json_artifact(name, _read(path))


def proof_stream(root, proof_id, name, start=0):
    directory = _proof_directory(root, proof_id)
    if not _ARTIFACT.fullmatch(name) or not name.endswith('.jsonl'):
        raise ValueError('Invalid stream name')
    path = directory / 'artifacts' / name
    if not path.exists():
        raise NotFound('Unknown stream')
    return stream_chunk(path, start)


# -- literature and referee reports ----------------------------------------

def research_state(root, job_id):
    if not isinstance(job_id, str) or not _ID.fullmatch(job_id):
        raise ValueError('Invalid research ID')
    if not (Path(root) / '.mathagent' / 'research' / job_id).exists():
        raise NotFound('Unknown research job')
    return _read_research(_job_directory(root, job_id), job_id)


def list_research(root, active=None):
    root = Path(root)
    base = root / '.mathagent' / 'research'
    if not base.is_dir() or base.is_symlink():
        return []
    jobs = []
    for path in base.iterdir():
        if not _ID.fullmatch(path.name) or path.is_symlink():
            continue
        try:
            state = _cached(path / 'state.json', lambda: research_state(root, path.name))
        except (OSError, ValueError, KeyError, NotFound):
            continue
        running = state['status'] == 'running' and (path.name == active or locked(path))
        jobs.append({'id': state['id'], 'kind': state['kind'], 'status': state['status'],
                     'phase': state['phase'], 'title': one_line(state['goal'], 140),
                     'updated_at': state['updated_at'], 'created_at': state['created_at'],
                     'running': running, 'sources': [s['path'] for s in state['sources']],
                     'tokens': _budget(state['tokens_charged'], state['settings']['max_tokens'])})
    return sorted(jobs, key=lambda job: job['updated_at'], reverse=True)


def research_detail(root, job_id, active=None):
    s = research_state(root, job_id)
    directory = _job_directory(root, job_id)
    running = s['status'] == 'running' and (job_id == active or locked(directory))
    call = next((c for c in reversed(s['calls']) if c.get('status') == 'running'), None) if running else None
    settings = s['settings']
    usage = (s.get('literature_snapshot') or {}).get('stats', {})
    report = directory / 'report.md'
    return {
        'id': s['id'], 'kind': s['kind'], 'goal': s['goal'], 'status': s['status'], 'phase': s['phase'],
        'stop_reason': s.get('stop_reason'), 'created_at': s['created_at'], 'updated_at': s['updated_at'],
        'settings': settings,
        'budget': {'rounds': _budget(s['rounds_started'], settings['max_rounds']),
                   'tokens': _budget(s['tokens_charged'], settings['max_tokens']),
                   'input_tokens': _budget(s['input_tokens_charged'], settings['max_input_tokens']),
                   'seconds': _budget(round(s['seconds_used'], 1), settings['max_seconds']),
                   'requests': _budget(usage.get('requests', 0), settings['max_requests']),
                   'chars': _budget(usage.get('returned_chars', 0), settings['max_chars'])},
        'sources': [{'id': src['id'], 'path': src['path'], 'sha256': src['sha256'],
                     'lines': len(src['content'].splitlines())} for src in s['sources']],
        'plan': s['plan'], 'notes': s['notes'], 'evidence': s['evidence'], 'draft': s['draft'],
        'review': s['review'], 'draft_complete': s.get('draft_complete'),
        'review_complete': s.get('review_complete'), 'warnings': s.get('warnings', []),
        'citation_issues': s.get('citation_issues', []),
        'review_context_issues': s.get('review_context_issues', []),
        'manuscript_ranges': s.get('manuscript_ranges', {}), 'calls': s['calls'],
        'report': _read(report) if report.exists() else '', 'running': running,
        'live': {'file': call['stream'], 'role': call['role']} if call else None,
        'artifacts': _artifacts(directory, _RESEARCH_ARTIFACT),
        **({key: s.get(key) for key in ('outline', 'sections', 'document', 'section_lines', 'compile',
                                        'outputs', 'macros', 'theorems', 'output_name')}
           if s['kind'] == 'writeup' else {}),
    }


def research_pdf(root, job_id):
    research_state(root, job_id)
    path = _job_directory(root, job_id) / 'artifacts' / 'compiled.pdf'
    if not path.exists():
        raise NotFound('This write-up has no compiled PDF')
    _regular(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        return stream.read(UPLOAD_BYTES)


def research_sources(root, job_id):
    return [{'id': src['id'], 'path': src['path'], 'sha256': src['sha256'], 'content': src['content']}
            for src in research_state(root, job_id)['sources']]


def research_artifact(root, job_id, name, stream_from=None):
    research_state(root, job_id)
    if not _RESEARCH_ARTIFACT.fullmatch(name):
        raise ValueError('Invalid artifact name')
    path = _job_directory(root, job_id) / 'artifacts' / name
    if not path.exists():
        raise NotFound('Unknown artifact')
    if stream_from is not None:
        return stream_chunk(path, stream_from)
    return _json_artifact(name, _read(path))


# -- workspace files that can be pinned ------------------------------------

def list_files(root, limit=2000):
    root = Path(root)
    found = []
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in SKIP
                         and not (Path(folder) / d).is_symlink())
        for name in sorted(files):
            path = Path(folder) / name
            suffix = path.suffix.lower()
            if name.startswith('.') or path.is_symlink() or (suffix not in TEXT and suffix != '.pdf'):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            found.append({'path': path.relative_to(root).as_posix(), 'size': size,
                          'kind': 'pdf' if suffix == '.pdf' else 'text'})
            if len(found) >= limit:
                return found
    return found


# -- chats --------------------------------------------------------------------

def _locked_chat(function):
    def wrapper(*args, **kwargs):
        with _chat_lock:
            return function(*args, **kwargs)
    wrapper.__name__, wrapper.__doc__ = function.__name__, function.__doc__
    return wrapper


def _chats(root, create=False):
    base = Path(root) / '.mathagent'
    if not create and not (base / 'chats').exists():
        return None
    _directory(base, create=create)
    _directory(base / 'chats', create=create)
    return base / 'chats'


def _read_chat(path):
    _regular(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('Refusing non-regular chat file')
        data = stream.read(CHAT_BYTES + 1)
    if len(data) > CHAT_BYTES:
        raise ValueError('Chat file is too large')
    return json.loads(data.decode('utf-8'))


def _save_chat(root, chat):
    chat['updated_at'] = _now()
    data = json.dumps(chat, ensure_ascii=False, indent=1).encode('utf-8')
    if len(data) > CHAT_BYTES:
        raise ValueError('This chat reached the 16 MB storage limit; start a new chat')
    _atomic_write(_chats(root, create=True) / (chat['id'] + '.json'), data)


def create_chat(root, mode, think=True, model='', online=False):
    if mode not in CHAT_MODES:
        raise ValueError('Chat mode must be critic, explore or free')
    now = _now()
    chat = {'version': 1, 'id': str(uuid.uuid4()), 'mode': mode, 'title': '',
            'created_at': now, 'updated_at': now,
            'settings': {'think': bool(think), 'model': model, 'online': bool(online)},
            'history': [], 'transcript': []}
    _save_chat(root, chat)
    return chat


def load_chat(root, chat_id):
    if not isinstance(chat_id, str) or not _ID.fullmatch(chat_id):
        raise ValueError('Invalid chat ID')
    base = _chats(root)
    if base is None or not (base / (chat_id + '.json')).exists():
        raise NotFound('Unknown chat')
    chat = _read_chat(base / (chat_id + '.json'))
    if (not isinstance(chat, dict) or chat.get('version') != 1 or chat.get('id') != chat_id
            or chat.get('mode') not in CHAT_MODES or not isinstance(chat.get('history'), list)
            or not isinstance(chat.get('transcript'), list) or not isinstance(chat.get('settings'), dict)):
        raise ValueError('Invalid or unsupported chat file')
    # Preserve the maximum budget of route cards saved before the effort rename.
    for item in chat['transcript']:
        if isinstance(item, dict) and item.get('role') == 'route' and item.get('effort') == 'brezis':
            item['effort'] = 'poincare'
    return chat


@_locked_chat
def update_chat_settings(root, chat_id, *, think=None, online=None):
    chat = load_chat(root, chat_id)
    if think is not None:
        chat['settings']['think'] = bool(think)
    if online is not None:
        chat['settings']['online'] = bool(online)
    _save_chat(root, chat)
    return chat


def list_chats(root):
    base = _chats(root)
    if base is None:
        return []
    chats = []
    for path in base.iterdir():
        if not path.name.endswith('.json') or not _ID.fullmatch(path.name[:-5]) or path.is_symlink():
            continue
        try:
            chat = _cached(path, lambda: load_chat(root, path.name[:-5]))
        except (OSError, ValueError, NotFound):
            continue
        last = next((item['content'] if item.get('role') != 'route' else
                     f"{ROUTE_NAMES.get(item.get('mode'), 'Suggestion')}: {item.get('request', '')}"
                     for item in reversed(chat['transcript'])
                     if (item.get('role') in ('assistant', 'review') and item.get('content'))
                     or (item.get('role') == 'route' and item.get('status') not in ('dismissed',))), '')
        chats.append({'id': chat['id'], 'kind': chat['mode'], 'title': chat['title'] or 'New conversation',
                      'updated_at': chat['updated_at'], 'created_at': chat['created_at'],
                      'messages': sum(1 for item in chat['transcript'] if item.get('role') == 'user'),
                      'preview': one_line(last, 140)})
    return sorted(chats, key=lambda chat: chat['updated_at'], reverse=True)


@_locked_chat
def commit_turn(root, chat_id, history, calls, notices, echo_user=True, mode=None):
    """Save a completed turn: exact model context plus an append-only transcript."""
    chat = load_chat(root, chat_id)
    start = max(i for i, m in enumerate(history) if m.get('role') == 'user')
    stamp, index, items = _now(), 0, []
    for message in history[start:]:
        if message['role'] == 'user' and not echo_user:
            continue  # Default mode already shows the message and its route card
        item = {'role': message['role'], 'content': message.get('content', ''), 'time': stamp}
        if mode and message['role'] != 'user':
            item['mode'] = mode  # which workflow answered (Default mode mixes them)
        if message['role'] == 'assistant':
            if message.get('tool_calls'):
                item['tool_calls'] = message['tool_calls']
            if index < len(calls):
                item['thinking'], item['stats'] = calls[index]['thinking'], calls[index]['stats']
            index += 1
        elif message['role'] == 'tool':
            item['tool_name'] = message.get('tool_name', '')
        items.append(item)
    if notices:
        items.append({'role': 'notice', 'content': '\n'.join(dict.fromkeys(notices)), 'time': stamp})
    # The model context may drop old turns to fit; the transcript never does.
    chat['history'] = history
    chat['transcript'].extend(items)
    if not chat['title']:
        chat['title'] = one_line(history[start]['content'], 80)
    _save_chat(root, chat)


@_locked_chat
def commit_review(root, chat_id, result, calls, notices):
    chat = load_chat(root, chat_id)
    # Same history record as the terminal's /review.
    chat['history'].extend([{'role': 'user', 'content': 'Critique the preceding answer.'},
                            {'role': 'assistant', 'content': result}])
    item = {'role': 'review', 'content': result, 'time': _now(),
            'thinking': ''.join(call['thinking'] for call in calls),
            'stats': calls[-1]['stats'] if calls else {}}
    chat['transcript'].append(item)
    if notices:
        chat['transcript'].append({'role': 'notice', 'content': '\n'.join(dict.fromkeys(notices)), 'time': item['time']})
    _save_chat(root, chat)


@_locked_chat
def record_discarded(root, chat_id, content, reason):
    """Keep an unfinished turn visible without adding it to the model context."""
    chat = load_chat(root, chat_id)
    stamp = _now()
    if content:
        chat['transcript'].append({'role': 'user', 'content': content, 'time': stamp, 'discarded': True})
        if not chat['title']:
            chat['title'] = one_line(content, 80)
    chat['transcript'].append({'role': 'notice', 'content': 'Not added to the conversation: ' + reason,
                               'time': stamp, 'discarded': True})
    _save_chat(root, chat)


@_locked_chat
def append_items(root, chat_id, items):
    """Append display items (Default mode messages and route cards); returns the first index."""
    chat = load_chat(root, chat_id)
    index = len(chat['transcript'])
    chat['transcript'].extend(items)
    if not chat['title']:
        first = next((item['content'] for item in items if item.get('role') == 'user'), '')
        chat['title'] = one_line(first, 80) if first else ''
    _save_chat(root, chat)
    return index


@_locked_chat
def update_item(root, chat_id, index, **fields):
    chat = load_chat(root, chat_id)
    if type(index) is not int or not 0 <= index < len(chat['transcript']):
        raise NotFound('Unknown conversation item')
    chat['transcript'][index].update(fields)
    _save_chat(root, chat)
    return chat['transcript'][index]


# -- workspace settings, folders, uploads and templates -------------------------------

def _private_json(path):
    if not path.exists():
        return None
    try:
        value = json.loads(_read(path))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def load_settings(workspace):
    value = _private_json(Path(workspace) / '.mathagent' / 'gui-settings.json') or {}
    saved = value.get('read') if isinstance(value.get('read'), dict) else {}
    return {'read': {group: saved.get(group) is not False for group in READ_GROUPS}}


def save_settings(workspace, read):
    if not isinstance(read, dict) or any(k not in READ_GROUPS or type(v) is not bool for k, v in read.items()):
        raise ValueError('read must map tex, pdf, py and text to true or false')
    settings = load_settings(workspace)
    settings['read'].update(read)
    base = Path(workspace) / '.mathagent'
    _directory(base, create=True)
    _atomic_write(base / 'gui-settings.json', json.dumps(settings, indent=1).encode())
    return settings


def read_types(workspace):
    return [group for group, allowed in load_settings(workspace)['read'].items() if allowed]


def resolve_folder(root, relative=''):
    """A folder at or below the interface root: no hidden parts, no symlink escape."""
    root = Path(root).resolve(strict=True)
    if not isinstance(relative, str) or len(relative) > 1000:
        raise ValueError('Folder path must be text')
    rel = Path(relative.strip('/')) if relative.strip('/') else Path('.')
    if rel.is_absolute() or any(part == '..' or (part.startswith('.') and part != '.') for part in rel.parts):
        raise ValueError('Folders are relative to the interface root; hidden folders are excluded')
    target = (root / rel).resolve()
    if not target.is_relative_to(root) or any(part.startswith('.') for part in target.relative_to(root).parts):
        raise ValueError('That folder is outside the interface root')
    if not target.is_dir():
        raise NotFound('No such folder')
    return target


def _counts(folder):
    counts = {'tex': 0, 'pdf': 0, 'py': 0}
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return counts
    for entry in entries[:5000]:
        if entry.name.startswith('.') or not entry.is_file(follow_symlinks=False):
            continue
        suffix = Path(entry.name).suffix.lower()
        key = 'tex' if suffix in READ_GROUPS['tex'] else 'pdf' if suffix == '.pdf' else 'py' if suffix == '.py' else None
        if key:
            counts[key] += 1
    return counts


def list_folders(root, relative=''):
    root = Path(root).resolve(strict=True)
    target = resolve_folder(root, relative)
    folders = []
    for entry in sorted(os.scandir(target), key=lambda e: e.name.lower()):
        if (entry.name.startswith('.') or entry.name in SKIP or entry.is_symlink()
                or not entry.is_dir(follow_symlinks=False)):
            continue
        path = Path(entry.path)
        folders.append({'name': entry.name, 'path': path.relative_to(root).as_posix(), 'files': _counts(path),
                        'jobs': (path / '.mathagent' / 'proofs').is_dir() or (path / '.mathagent' / 'research').is_dir()})
        if len(folders) >= 500:
            break
    rel = target.relative_to(root).as_posix()
    return {'root': str(root), 'path': '' if rel == '.' else rel, 'files': _counts(target), 'folders': folders}


def create_folder(root, relative, name):
    parent = resolve_folder(root, relative)
    if (not isinstance(name, str) or not name.strip() or len(name) > 100 or name.strip().startswith('.')
            or any(c in name for c in '/\\\0')):
        raise ValueError('Folder names cannot be empty, hidden, or contain slashes')
    target = parent / name.strip()
    target.mkdir()
    return target.relative_to(Path(root).resolve()).as_posix()


def remember_folder(root, relative):
    base = Path(root) / '.mathagent'
    _directory(base, create=True)
    state = _private_json(base / 'gui-state.json') or {}
    recent = [relative] + [r for r in state.get('recent', []) if isinstance(r, str) and r != relative]
    state['recent'] = recent[:8]
    _atomic_write(base / 'gui-state.json', json.dumps(state, indent=1).encode())


def recent_folders(root):
    state = _private_json(Path(root) / '.mathagent' / 'gui-state.json') or {}
    return [r for r in state.get('recent', []) if isinstance(r, str)][:8]


def save_upload(workspace, name, data):
    """Save a dropped file in the workspace; never replaces an existing file."""
    workspace = Path(workspace).resolve(strict=True)
    base = re.sub(r'[^\w.\- ()+]', '_', str(name).replace('\\', '/').split('/')[-1]).strip()
    suffix = Path(base).suffix.lower()
    if not base or base.startswith('.') or suffix not in UPLOAD_SUFFIXES:
        raise ValueError('Drop .tex, .sty, .cls, .bib, .md, .txt, .pdf or .py files')
    if len(data) > UPLOAD_BYTES:
        raise ValueError('Files are limited to 20 MB')
    if suffix != '.pdf' and b'\x00' in data:
        raise ValueError('That file is not text')
    stem, number = Path(base).stem, 2
    target = workspace / base
    fd, temporary = tempfile.mkstemp(prefix='.upload-', dir=workspace)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        while True:
            try:
                os.link(temporary, target)  # link() never replaces an existing name
                break
            except FileExistsError:
                target = workspace / f'{stem}-{number}{suffix}'
                number += 1
    finally:
        os.unlink(temporary)
    return target.name


def _template_dir(root, name):
    if not isinstance(name, str) or not re.fullmatch(r'[\w][\w .-]{0,59}', name):
        raise ValueError('Template names use letters, digits, spaces, dots and dashes (at most 60)')
    return Path(root) / '.mathagent' / 'templates' / name


def list_templates(root):
    base = Path(root) / '.mathagent' / 'templates'
    if not base.is_dir() or base.is_symlink():
        return []
    found = []
    for folder in sorted(base.iterdir()):
        if folder.is_dir() and not folder.is_symlink():
            files = sorted(p.name for p in folder.iterdir() if p.is_file() and not p.is_symlink()
                           and p.suffix.lower() in TEMPLATE_SUFFIXES)
            found.append({'name': folder.name, 'files': files})
    return found


def save_template(root, workspace_obj, name, paths):
    """Copy template files from the workspace so they can be reused in any folder."""
    target = _template_dir(root, name)
    contents = []
    for path in paths:
        p = workspace_obj.path(path)
        if p.suffix.lower() not in TEMPLATE_SUFFIXES:
            raise ValueError('Template files must be .sty, .cls, .tex or .bib')
        contents.append((p.name, workspace_obj.text(p)))
    if not contents:
        raise ValueError('Choose the template files to remember')
    _directory(Path(root) / '.mathagent', create=True)
    _directory(target.parent, create=True)
    target.mkdir(exist_ok=True)
    for old in target.iterdir():
        if old.is_file() and not old.is_symlink():
            old.unlink()
    for file_name, content in contents:
        _atomic_write(target / file_name, content.encode('utf-8'))
    return {'name': name, 'files': [n for n, _ in contents]}


def load_template(root, name):
    folder = _template_dir(root, name)
    if not folder.is_dir() or folder.is_symlink():
        raise NotFound('Unknown template')
    return [(p.name, _read(p)) for p in sorted(folder.iterdir())
            if p.is_file() and not p.is_symlink() and p.suffix.lower() in TEMPLATE_SUFFIXES]
