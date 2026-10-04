"""Structured failures that cannot be repaired by increasing model effort."""
import copy
import json


class WorkerInputError(ValueError):
    def __init__(self, message, *, code='invalid_worker_input', details=None, scope='input'):
        super().__init__(message)
        self.code, self.details, self.scope = code, copy.deepcopy(details or {}), scope
        json.dumps(self.details, allow_nan=False)

    def as_dict(self):
        return {'code': self.code, 'message': str(self), 'retryable': False,
                'scope': self.scope, 'details': copy.deepcopy(self.details)}
