"""Guess which workflow a free-form request needs, and restate it self-contained.

One short structured call to the same local model. The guess is shown to the
user before anything runs: it is a suggestion, never a permission, and it
cannot change network, Python or file-access settings.
"""
import json

JOB_MODES = ('prove', 'critic', 'explore', 'literature', 'referee', 'writeup')
ROUTE_MODES = JOB_MODES + ('clarify',)

ROUTE_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['mode', 'request', 'files', 'reason', 'question'],
    'properties': {
        'mode': {'type': 'string', 'enum': list(ROUTE_MODES)},
        'request': {'type': 'string', 'maxLength': 8000},
        'files': {'type': 'array', 'maxItems': 20, 'items': {'type': 'string', 'maxLength': 300}},
        'reason': {'type': 'string', 'maxLength': 400},
        'question': {'type': 'string', 'maxLength': 600},
    },
}

ROUTER_POLICY = """You route a mathematician's message to one workflow of a local research harness.
- prove: a bounded proof search for one precise mathematical statement to be proved.
- critic: check a given argument, proof or claim for gaps and counterexamples, or answer a
  precise mathematical question that needs a justified answer.
- explore: open discussion of approaches, ideas, background or connections.
- literature: a bibliographical report comparing published results on a topic.
- referee: a referee report assessing a manuscript file.
- writeup: turn notes, a draft or a PDF into a clean LaTeX document.
- clarify: the message is too ambiguous to choose; ask one short question.
Rewrite `request` so that it is complete on its own. Proof and report jobs never see the
conversation: copy every hypothesis, definition, statement and file name they need from the
context. Keep the user's language. Do not solve or answer anything yourself.
`files` lists workspace files the task needs: the user's attachments and files named in the
message or context, spelled exactly as listed. `reason` explains the choice in one sentence.
`question` is empty unless the mode is clarify. Return only JSON matching the schema."""

FALLBACK_QUESTION = ('Should I prove a statement, check an argument, explore an idea, write a '
                     'literature or referee report, or write up notes in LaTeX?')


def parse(text, message, files=()):
    """Validate a routing answer; anything malformed becomes a clarifying question."""
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if not isinstance(value, dict) or value.get('mode') not in ROUTE_MODES:
        return {'mode': 'clarify', 'request': message, 'files': list(files),
                'reason': 'The routing answer was not usable.', 'question': FALLBACK_QUESTION}
    request = value.get('request') if isinstance(value.get('request'), str) else ''
    proposed = [f for f in value.get('files') or [] if isinstance(f, str) and f.strip()]
    route = {
        'mode': value['mode'],
        'request': (request.strip() or message)[:8000],
        'files': list(dict.fromkeys([*files, *proposed]))[:20],
        'reason': str(value.get('reason') or '')[:400],
        'question': str(value.get('question') or '')[:600],
    }
    if route['mode'] == 'clarify' and not route['question'].strip():
        route['question'] = FALLBACK_QUESTION
    return route


def classify(client, model, message, *, context='', files=(), available=(), ctx=8192):
    """One structured call (thinking off, about 700 tokens) returning a route."""
    room = max(2000, (ctx - 700 - 400) * 3 - len(ROUTER_POLICY) - len(message) - 2000)
    parts = []
    if context:
        parts.append('EARLIER IN THIS CONVERSATION (oldest first):\n' + context[-room:])
    if available:
        parts.append('WORKSPACE FILES:\n' + '\n'.join(list(available)[:200]))
    if files:
        parts.append('ATTACHED BY THE USER:\n' + '\n'.join(files))
    parts.append('MESSAGE:\n' + message)
    payload = {'model': model, 'stream': True, 'think': False, 'format': ROUTE_SCHEMA,
               'options': {'num_ctx': ctx, 'num_predict': 700, 'temperature': 0},
               'messages': [{'role': 'system', 'content': ROUTER_POLICY},
                            {'role': 'user', 'content': '\n\n'.join(parts)}]}
    text = ''
    for event in client.stream(payload):
        content = (event.get('message') or {}).get('content')
        if isinstance(content, str):
            text += content
    return parse(text, message, files)
