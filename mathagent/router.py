"""Guess which workflows a free-form request needs, and restate each self-contained.

One short structured call to the same local model. The guess is shown to the
user before anything runs: it is a suggestion, never a permission, and it
cannot change network, Python or file-access settings. A message may ask for
several separate jobs; each becomes its own suggestion with a suggested effort.
"""
import json
import re

JOB_MODES = ('prove', 'critic', 'explore', 'check', 'literature', 'referee', 'writeup')
ROUTE_MODES = JOB_MODES + ('clarify',)
EFFORTS = ('low', 'medium', 'high', 'xhigh', 'poincare')
MAX_TASKS = 4
ROUTE_PREDICT = 1600

TASK_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['mode', 'request', 'files', 'effort', 'reason'],
    'properties': {
        'mode': {'type': 'string', 'enum': list(JOB_MODES)},
        'request': {'type': 'string', 'maxLength': 8000},
        'files': {'type': 'array', 'maxItems': 20, 'items': {'type': 'string', 'maxLength': 300}},
        'effort': {'type': 'string', 'enum': list(EFFORTS)},
        'reason': {'type': 'string', 'maxLength': 400},
    },
}

ROUTE_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['tasks', 'question'],
    'properties': {
        'tasks': {'type': 'array', 'maxItems': MAX_TASKS, 'items': TASK_SCHEMA},
        'question': {'type': 'string', 'maxLength': 600},
    },
}

ROUTER_POLICY = """You route a mathematician's message to the workflows of a local research harness.
- critic: answer a precise mathematical question that needs a justified answer, or check a
  given argument, proof or claim for gaps and counterexamples.
- explore: open discussion of approaches, ideas, background or connections.
- prove: a bounded proof search for one precise mathematical statement to be proved.
- check: find a useful standard book/paper reference for a known result, optionally a chapter,
  or verify a given citation or an explicitly requested exact locator. Ordinary reference requests
  need one bounded lookup and an honest qualified suggestion, not a mandatory theorem number.
  Do not add exact section/theorem/page requirements the human did not ask for.
  Examples: "find a reference for elliptic regularity
  with Neumann conditions", "where is the Rellich theorem proved?", "is Theorem 9.26 of Brezis about this?".
- literature: a reading list or map of the literature on a topic (what to read, surveys, core papers by
  theme, recent developments), e.g. "what should I read on…", "survey the literature on…".
- referee: a review report assessing a manuscript file the user names or attaches.
- writeup: turn notes, a draft or a PDF the user names or attaches into a clean LaTeX document.
A question (why, what, how, is it true that) is answered with critic or explore. A request for a
reference or citation to a known result is a check, not a literature report. Choose
prove, literature, referee or writeup only when the message asks for that kind of work;
never choose one just because files exist in the workspace.
Return one task per separate piece of work, in the order they should run. A single question,
statement or request is exactly one task. Give several (at most 4) only when the message
explicitly asks for different jobs, such as "prove this lemma and also survey its literature".
For each task, rewrite `request` so that it is complete on its own. Proof and report jobs
never see the conversation: copy every hypothesis, definition, statement and file name they
need from the context. Keep the user's language. Do not solve or answer anything yourself.
`files` lists workspace files the task needs: the user's attachments and files named in the
message or context, spelled exactly as listed.
`effort` sizes the job: low for a quick check or a routine statement, medium for ordinary
work, high for a hard problem or a long manuscript, xhigh for research-level problems, and
poincare only when the user asks for the maximum effort.
`reason` explains the choice in one sentence.
When the message is too ambiguous to choose, return no tasks and ask one short `question`;
otherwise `question` is empty. Return only JSON matching the schema."""

FALLBACK_QUESTION = ('Should I prove a statement, check an argument, explore an idea, find a reference, '
                     'write a literature report or a review of a manuscript, or write up notes in LaTeX?')


# In JSON, "\frac" written with one backslash reads as a form feed and "rac"; small models
# often forget to double the backslash of TeX commands. Restore the command when a letter
# follows the control character (and, for a newline, only known commands).
_CONTROL = {'\f': 'f', '\b': 'b', '\t': 't', '\r': 'r'}
_AFTER_N = ('abla', 'eq', 'otin', 'ot', 'mid', 'geq', 'leq', 'ewline', 'oindent', 'subseteq')


def repair_latex(text):
    text = re.sub('[\f\b\t\r](?=[A-Za-z])', lambda m: '\\' + _CONTROL[m.group()], text)
    return re.sub('\n(?=(?:' + '|'.join(_AFTER_N) + r')\b)', r'\\n', text)


def clarify(message, files=(), question='', reason=''):
    """A suggestion that asks one question instead of starting anything."""
    return {'mode': 'clarify', 'request': message, 'files': list(files), 'effort': 'medium',
            'reason': reason, 'question': repair_latex((question or '').strip())[:600] or FALLBACK_QUESTION}


def _task(value, message, files):
    request = value.get('request') if isinstance(value.get('request'), str) else ''
    proposed = [f for f in value.get('files') or [] if isinstance(f, str) and f.strip()]
    return {
        'mode': value['mode'],
        'request': repair_latex(request.strip() or message)[:8000],
        'files': list(dict.fromkeys([*files, *proposed]))[:20],
        'effort': value.get('effort') if value.get('effort') in EFFORTS else 'medium',
        'reason': repair_latex(str(value.get('reason') or ''))[:400],
        'question': '',
    }


def parse(text, message, files=()):
    """Validate a routing answer into a list of suggestions.

    Anything malformed becomes one clarifying question. A single task object
    (the earlier answer shape) is accepted as a list of one.
    """
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if isinstance(value, dict) and 'tasks' not in value and 'mode' in value:
        if value.get('mode') == 'clarify':
            return [clarify(message, files, value.get('question'), str(value.get('reason') or '')[:400])]
        value = {'tasks': [value], 'question': ''}
    if not isinstance(value, dict) or not isinstance(value.get('tasks'), list):
        return [clarify(message, files, '', 'The routing answer was not usable.')]
    tasks = [_task(item, message, files) for item in value['tasks']
             if isinstance(item, dict) and item.get('mode') in JOB_MODES][:MAX_TASKS]
    if not tasks:
        question = value.get('question') if isinstance(value.get('question'), str) else ''
        return [clarify(message, files, question, '' if question.strip() else 'The routing answer was not usable.')]
    return tasks


def classify(client, model, message, *, context='', files=(), available=(), ctx=8192):
    """One structured call (thinking off, at most ROUTE_PREDICT tokens) returning suggestions."""
    room = max(2000, (ctx - ROUTE_PREDICT - 400) * 3 - len(ROUTER_POLICY) - len(message) - 2000)
    parts = []
    if context:
        parts.append('EARLIER IN THIS CONVERSATION (oldest first):\n' + context[-room:])
    if available:
        parts.append('WORKSPACE FILES (only to spell the names of files the message mentions; '
                     'never use a file the message does not mention):\n' + '\n'.join(list(available)[:200]))
    if files:
        parts.append('ATTACHED BY THE USER:\n' + '\n'.join(files))
    parts.append('MESSAGE:\n' + message)
    payload = {'model': model, 'stream': True, 'think': False, 'format': ROUTE_SCHEMA,
               'options': {'num_ctx': ctx, 'num_predict': ROUTE_PREDICT, 'temperature': 0},
               'messages': [{'role': 'system', 'content': ROUTER_POLICY},
                            {'role': 'user', 'content': '\n\n'.join(parts)}]}
    text = ''
    for event in client.stream(payload):
        content = (event.get('message') or {}).get('content')
        if isinstance(content, str):
            text += content
    return parse(text, message, files)
