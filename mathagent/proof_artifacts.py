"""Bounded, readable views of immutable proof artifacts.

Transport journals remain the forensic source of truth. These views decode
their written response without replaying tools, exposing model thinking, or
turning a model verdict into a controller-approved mathematical claim.
"""
import json
import re

from .proof_policy import (
    AUDIT_SCHEMA, CRITIC_SCHEMA, PLAN_SCHEMA, REVIEW_BATCH_SCHEMA, REVIEW_SCHEMA,
)


PAGE_CHARACTERS = 6000
MAX_PAGE_CHARACTERS = 7800
MAX_NORMALIZED_CHARACTERS = 100000
_BASENAME = re.compile(r'[0-9]{4,}-[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\.(?:md|jsonl)\Z')
_REVIEW_SCHEMAS = {
    'critic': CRITIC_SCHEMA,
    'recorder': REVIEW_BATCH_SCHEMA,
    'recorder-repair': REVIEW_BATCH_SCHEMA,
    'auditor': AUDIT_SCHEMA,
    'planner': PLAN_SCHEMA,
}


class _Unreadable(ValueError):
    pass


def is_artifact_basename(value):
    """Recognize only exact safe IDs, never filesystem paths or aliases."""
    return isinstance(value, str) and '..' not in value and bool(_BASENAME.fullmatch(value))


def _matches(value, spec):
    kind = spec['type']
    if kind == 'object':
        return (isinstance(value, dict) and set(value) == set(spec['properties'])
                and all(_matches(value[key], child) for key, child in spec['properties'].items()))
    if kind == 'array':
        return (isinstance(value, list) and len(value) <= spec['maxItems']
                and all(_matches(item, spec['items']) for item in value))
    if kind == 'string':
        return (isinstance(value, str) and len(value) <= spec.get('maxLength', 10000)
                and ('enum' not in spec or value in spec['enum']))
    return kind == 'boolean' and type(value) is bool


def _decode_json(text):
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        raise _Unreadable('The saved response is not valid JSON.') from None


def _stream_text(raw):
    fragments, size, finished = [], 0, False
    for line in raw.splitlines():
        if not line.strip():
            continue
        if finished:
            raise _Unreadable('The stream has data after its completion event.')
        event = _decode_json(line)
        if not isinstance(event, dict) or 'error' in event:
            raise _Unreadable('The stream contains an invalid event or a transport error.')
        if 'done' in event and type(event['done']) is not bool:
            raise _Unreadable('The stream has an invalid completion flag.')
        message = event.get('message', {})
        if not isinstance(message, dict):
            raise _Unreadable('The stream contains an invalid message.')
        content = message.get('content', '')
        if content is None:
            content = ''
        if not isinstance(content, str):
            raise _Unreadable('The stream contains non-text response content.')
        if message.get('tool_calls'):
            raise _Unreadable('This stream requested tools; it is not a standalone written response.')
        size += len(content)
        if size > MAX_NORMALIZED_CHARACTERS:
            raise _Unreadable('The written response exceeds the normalized-view limit.')
        fragments.append(content)
        if event.get('done'):
            if event['done'] is not True or event.get('done_reason') != 'stop':
                raise _Unreadable('The stream did not finish with a normal stop; no complete response is available.')
            finished = True
    if not finished:
        raise _Unreadable('The stream is partial: no normal completion event was saved.')
    return ''.join(fragments)


def _normalized(filename, raw):
    label = filename.split('-', 1)[1].rsplit('.', 1)[0]
    if filename.endswith('.jsonl'):
        written = _stream_text(raw)
        if label in _REVIEW_SCHEMAS:
            review = _decode_json(written)
            spec = _REVIEW_SCHEMAS[label]
            # Early saved runs used the single-record schema.
            legacy = label == 'recorder' and _matches(review, REVIEW_SCHEMA)
            if not legacy and not _matches(review, spec):
                raise _Unreadable('The saved review does not match its structured response schema.')
            return json.dumps(review, ensure_ascii=False, indent=2), 'review_json', True, (
                'Saved model review; this view does not certify the proof or replace ledger status.')
        return written, 'model_text', True, 'Saved model response, not a proof certificate.'
    if label in {'candidate', 'checkpoint'}:
        draft = _decode_json(raw)
        if (not isinstance(draft, dict) or not isinstance(draft.get('text'), str)
                or type(draft.get('complete')) is not bool):
            raise _Unreadable('The saved candidate envelope is invalid.')
        text = draft['text']
        if len(text) > MAX_NORMALIZED_CHARACTERS:
            raise _Unreadable('The written candidate exceeds the normalized-view limit.')
        if not text.strip():
            raise _Unreadable('No written candidate text was saved; raw reasoning is available only with raw=true.')
        complete = (draft['complete'] and not draft.get('from_truncated_attempt', False)
                    and draft.get('transport_complete', True) is True
                    and not draft.get('calls'))
        notice = ('Complete written response; mathematical correctness still requires review.' if complete else
                  'INCOMPLETE CANDIDATE / CHECKPOINT: partial work, not a complete proof submission.')
        if draft.get('material_clipped'):
            notice += ' The source attempt was excerpted when this checkpoint was produced.'
        if draft.get('calls'):
            notice += ' This response requested tools and is not a completed standalone candidate.'
        return text, 'candidate_text', complete, notice
    return raw, 'text', None, 'Saved artifact text; use raw=true for the unchanged storage representation.'


def read_artifact_page(store, filename, offset=0, *, raw=False):
    """Read through the store's path checks and return a capped JSON page.

Offsets address the chosen representation, not raw bytes. Invalid or partial
structured artifacts return a short diagnostic, never a truncated verdict.
The underlying file is never rewritten by normalization.
"""
    if type(offset) is not int or offset < 0:
        raise ValueError('offset must be a nonnegative integer')
    if type(raw) is not bool:
        raise ValueError('raw must be a boolean')
    source = store.read_artifact(filename)
    representation, complete = 'raw', None
    notice = 'Unchanged stored text, including transport metadata or reasoning where present.'
    text = source
    if not raw:
        try:
            text, representation, complete, notice = _normalized(filename, source)
        except _Unreadable as exc:
            text, representation, complete = '', 'unavailable', False
            notice = str(exc) + ' Use raw=true for paged diagnostics; do not infer a proof verdict from this artifact.'
    end = min(offset + PAGE_CHARACTERS, len(text))
    page = {
        'filename': filename, 'representation': representation,
        'response_complete': complete, 'notice': notice,
        'offset': offset, 'total_characters': len(text),
        'next_offset': end if end < len(text) else None,
        'content': text[offset:end],
    }
    encoded = json.dumps(page, ensure_ascii=False)
    # Quoting/backslashes may otherwise double a page's tool/context cost.
    # Keep valid JSON even when the generic read_file compatibility route is
    # subject to the controller's 8000-character tool-result guard.
    while len(encoded) > MAX_PAGE_CHARACTERS:
        end -= max(1, (len(encoded) - MAX_PAGE_CHARACTERS + 1) // 2)
        page['content'] = text[offset:end]
        page['next_offset'] = end if end < len(text) else None
        encoded = json.dumps(page, ensure_ascii=False)
    return encoded
