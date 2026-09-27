"""Proof mode cannot acquire chat tools or unrelated workspace material."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from mathagent.agent import Agent
from mathagent.literature import LiteratureTools
from mathagent.proof import ProofRunner
from mathagent.tools import Workspace


class OfflineProofTests(unittest.TestCase):
    def test_online_chat_settings_do_not_expose_tools_to_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'unrelated.txt').write_text('PRIVATE UNRELATED CONTENT')
            (root / 'statement.txt').write_text('For real x, prove x=x.')
            library = LiteratureTools(root, online=True)
            library.execute = Mock(side_effect=AssertionError('No literature in proof mode'))
            workspace = Workspace(root, literature=library)
            workspace.execute = Mock(side_effect=AssertionError('No tool execution in proof mode'))
            requests = []

            class Client:
                host = 'http://localhost:11434'
                def stream(self, payload):
                    requests.append(payload)
                    text = ('By reflexivity, x=x.' if len(requests) == 1 else json.dumps({
                        'verdict': 'no_issue_found', 'explanation': 'Reflexivity applies.', 'issues': []}))
                    yield {'message': {'content': text}, 'done': True,
                           'done_reason': 'stop', 'eval_count': 20}

            runner = ProofRunner(Agent(Client(), workspace, ctx=8192))
            result = runner.start('Prove the pinned statement.', source_files=['statement.txt'],
                                  max_predict=1024, verify_tokens=1024, max_tokens=4096)
            self.assertEqual(result['status'], 'candidate_complete')
            self.assertEqual(len(requests), 2)
            for request in requests:
                self.assertNotIn('tools', request)
                self.assertNotIn('PRIVATE UNRELATED CONTENT', str(request))
                self.assertIn('For real x, prove x=x.', str(request))
            library.execute.assert_not_called()
            workspace.execute.assert_not_called()

    def test_offline_workspace_never_offers_unsandboxed_python(self):
        with tempfile.TemporaryDirectory() as directory:
            library = LiteratureTools(directory, online=False)
            workspace = Workspace(directory, allow_python=True, literature=library)
            self.assertNotIn('run_python', {s['function']['name'] for s in workspace.schemas()})
            self.assertIn('disabled', workspace.execute('run_python', {'code': 'print(1)'}))


if __name__ == '__main__':
    unittest.main()
