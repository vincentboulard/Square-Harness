"""Explicitly selected GUI mode calls, without interpreting ordinary @ text."""
from .orchestrator import Orchestrator
from .router import JOB_MODES


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

    Ordinary turns use the unchanged controller. Explicit calls use its existing
    worker adapters, shared budget, transcript, cancellation and resume logic.
    """

    def __init__(self, *args, required_mode=None, **kwargs):
        if required_mode is not None and required_mode not in JOB_MODES:
            raise ValueError('Unknown selected mode call')
        self.required_mode = required_mode
        super().__init__(*args, **kwargs)

    def _new_state(self, query, files, history):
        super()._new_state(query, files, history)
        if self.required_mode is None:
            return
        action = {'mode': self.required_mode, 'request': query, 'files': list(files),
                  'effort': 'medium', 'reason': 'User explicitly selected @' + self.required_mode + '.'}
        self._validate_delegate(action)
        self.state['required_mode'] = self.required_mode
        self._prepare_calls({'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': self.state['id'] + '_mode_call', 'type': 'function',
             'function': {'name': 'delegate', 'arguments': action}}]})
