"""Research capabilities cannot silently enter an unaided saved proof job."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from mathagent.agent import Agent
from mathagent.ledger import ProofStore
from mathagent.literature import LiteratureTools, TOOL_NAMES
from mathagent.proof import ProofRunner
from mathagent.tools import Workspace


class OfflineProofTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.library = LiteratureTools(self.root, online=True)
        self.ws = Workspace(self.root, literature=self.library)
        self.runner = ProofRunner(Agent(Mock(), self.ws))
        self.runner.store = ProofStore.create(self.root, 'Prove x=x.',
            {'max_rounds': 1, 'max_tokens': 5000, 'max_seconds': 30})

    def test_legacy_and_default_proof_never_expose_or_dispatch_literature(self):
        self.library.execute = Mock(side_effect=AssertionError('No literature calls in unaided proof'))
        for allowed in (None, False):
            if allowed is not None:
                self.runner.state['settings']['allow_literature'] = allowed
            names = {item['function']['name'] for item in self.runner._tools()}
            self.assertFalse(names & TOOL_NAMES)
            self.assertIn('disabled', self.runner._tool('search_papers', {'query': 'reflexivity'}))
        self.library.execute.assert_not_called()

    def test_explicit_assisted_proof_can_use_tools_but_offline_still_blocks_fetch(self):
        self.runner.state['settings']['allow_literature'] = True
        self.library.online = False
        self.library._fetch = Mock(side_effect=AssertionError('Offline must not fetch'))
        names = {item['function']['name'] for item in self.runner._tools()}
        self.assertIn('search_papers', names)
        self.assertIn('Offline', self.runner._tool('search_papers', {'query': 'reflexivity'}))
        self.library._fetch.assert_not_called()

    def test_offline_workspace_does_not_offer_arbitrary_python_even_if_enabled_programmatically(self):
        self.ws.allow_python = True
        self.library.online = False
        self.assertNotIn('run_python', {s['function']['name'] for s in self.ws.schemas()})
        self.assertIn('disabled', self.ws.execute('run_python', {'code': 'print(1)'}))

    def test_proof_budget_snapshot_is_saved_before_network_side_effect(self):
        self.runner.state['settings']['allow_literature'] = True
        self.library.on_budget_change = self.runner._save
        self.library.stats['requests'] = 2
        self.library.on_budget_change()
        saved = ProofStore.load(self.root, self.runner.state['id']).state
        self.assertEqual(saved['literature']['stats']['requests'], 2)
        self.assertNotIn('API_KEY', str(saved))


if __name__ == '__main__':
    unittest.main()
