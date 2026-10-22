"""Durable explicit mode calls and conservative automatic source/reference checks."""
import re

from .orchestrator import Orchestrator
from .router import JOB_MODES
from .source_tasks import automatic_source_action, _instruction_text
from .refcheck import standalone_reference_request


_NO_SEARCH = re.compile(
    r"\b(?:from memory|memory only|no search|(?:do not|don.t) (?:search|check|verify|browse)|"
    r'without (?:searching|checking|browsing)|without looking (?:it up|online)|'
    r'no (?:online|web) (?:search|lookup)|sans (?:chercher|verifier|recherche)|de memoire)\b')
_SEARCH_ALLOWED = re.compile(
    r'\b(?:(?:please )?(?:search|check|verify|browse) (?:online|the web|the internet)|'
    r'(?:use|enable|allow) (?:online|web) search|you (?:can|may) (?:search|browse)|'
    r'cherche(?:z)? (?:en ligne|sur internet))\b')


_REFERENCE_FOLLOWUP = re.compile(
    r'^(?:what about|how about|and (?:for|on|about|in)|et (?:pour|sur|en)|'
    r'qu.en est.il|et avec)\b')


def _reference_followup(text, history):
    if (not _REFERENCE_FOLLOWUP.match(text.strip())
            or re.search(r'\b(?:prove|derive|compute|calculate|explain|verify|check|audit|'
                         r'proof|proofs|counterexample|counterexamples|proving|explaining|'
                         r'prouve\w*|demontre\w*|preuve\w*|contre.exemple\w*|explique\w*|verifie\w*)\b', text)):
        return False
    for item in reversed(history):
        if not isinstance(item, dict) or item.get('role') != 'user':
            continue
        previous = _instruction_text(item.get('content', ''))
        if standalone_reference_request(previous):
            return True
        if not _REFERENCE_FOLLOWUP.match(previous.strip()):
            return False
    return False


def automatic_reference_action(query, files=(), history=()):
    """Seed a bounded check for an explicit, standalone reference request."""
    if (files or not isinstance(query, str) or len(query) > 8000
            or re.search(r'https?://|www\.|(?<![\w.])\S+\.(?:pdf|tex|md|txt)\b', query, re.I)):
        return None
    text = _instruction_text(query)
    if not (standalone_reference_request(text) or _reference_followup(text, history)):
        return None
    search_allowed = True
    human_texts = [item.get('content', '') for item in history
                   if isinstance(item, dict) and item.get('role') == 'user'] + [query]
    for human in human_texts:
        instruction = _instruction_text(human)
        if _NO_SEARCH.search(instruction):
            search_allowed = False
        elif _SEARCH_ALLOWED.search(instruction):
            search_allowed = True
    if not search_allowed:
        return None
    return {'mode': 'check', 'request': query, 'files': [], 'effort': 'low',
            'reason': 'Find an appropriate scholarly reference for the exact request and verify its bibliographic metadata.'}


def _word(char):
    return char.isalnum() or char in '_@.+/-'


def validated_mode_call(content, selection):
    if selection is None:
        return content, None
    if not isinstance(selection, dict) or set(selection) != {'mode', 'start'}:
        raise ValueError('A selected mode call requires mode and start')
    mode, start = selection['mode'], selection['start']
    if not isinstance(mode, str) or mode not in JOB_MODES or type(start) is not int:
        raise ValueError('Unknown selected mode call')
    token = '@' + mode
    end = start + len(token)
    if (start < 0 or content[start:end] != token
            or (start and _word(content[start - 1]))
            or (end < len(content) and _word(content[end]))):
        raise ValueError('The selected mode call no longer matches the message')
    query = (content[:start] + content[end:]).strip()
    if not query:
        raise ValueError('Add a request after selecting a mode')
    if len(query) > 8000:
        raise ValueError('Mode-call requests are limited to 8000 characters; attach longer material as a file')
    return query, mode


class ModeOrchestrator(Orchestrator):
    """Seed one durable delegation before the first main-agent call.

    Explicit modes and narrow source/reference requests use the existing
    worker adapters, shared budget, transcript, cancellation and resume logic.
    """

    def __init__(self, *args, required_mode=None, **kwargs):
        if required_mode is not None and required_mode not in JOB_MODES:
            raise ValueError('Unknown selected mode call')
        self.required_mode = required_mode
        super().__init__(*args, **kwargs)

    def _new_state(self, query, files, history):
        super()._new_state(query, files, history)
        action = ({'mode': self.required_mode, 'request': query, 'files': list(files),
                   'effort': 'medium', 'reason': 'User explicitly selected @' + self.required_mode + '.'}
                  if self.required_mode is not None else
                  automatic_source_action(query, files) or automatic_reference_action(query, files, history))
        if action is None:
            return
        self._validate_delegate(action)
        if self.required_mode is not None:
            self.state['required_mode'] = self.required_mode
        else:
            self.state['automatic_source_mode'] = action['mode']
            if action['mode'] == 'check':
                self.state['automatic_reference_mode'] = True
        self._prepare_calls({'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': self.state['id'] + '_mode_call', 'type': 'function',
             'function': {'name': 'delegate', 'arguments': action}}]})
