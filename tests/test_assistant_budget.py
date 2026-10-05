import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from mathagent.assistant_budget import AssistantBudget, BudgetClient


class Client:
    timeout = 10

    def __init__(self, used=12):
        self.used = used

    def count_input_tokens(self, payload):
        return 20

    def stream(self, payload):
        yield {'done': True, 'eval_count': self.used, 'prompt_eval_count': 20}


def payload(cap=40):
    return {'messages': [{'role': 'user', 'content': 'Exact statement.'}],
            'options': {'num_predict': cap, 'num_ctx': 2048}}


class AssistantBudgetTests(unittest.TestCase):
    def test_adaptive_promotion_preserves_spending_and_saved_caps_on_resume(self):
        budget = BudgetClient(Client(), max_tokens=100, max_input_tokens=100, max_seconds=10,
                              adaptive_limits={'max_tokens': 200, 'max_seconds': 20})
        list(budget.stream(payload()))
        budget.ensure_limits(max_tokens=300, max_input_tokens=300, max_seconds=30)
        self.assertEqual((budget.snapshot()['tokens'], budget.snapshot()['input_tokens']), (12, 20))
        self.assertEqual((budget.snapshot()['max_tokens'], budget.snapshot()['max_input_tokens'], budget.snapshot()['max_seconds']),
                         (200, 100, 20))
        resumed = BudgetClient(Client(), state=budget.snapshot(), adaptive_limits={'max_tokens': 1000})
        resumed.ensure_limits(max_tokens=1000)
        self.assertEqual(resumed.snapshot()['max_tokens'], 200)
        self.assertEqual(resumed.remaining_tokens, 188)

    def test_legacy_saved_budget_cannot_be_promoted_on_resume(self):
        old = BudgetClient(Client(), max_tokens=100).snapshot()
        resumed = BudgetClient(Client(), state=old, adaptive_limits={'max_tokens': 1000})
        resumed.ensure_limits(max_tokens=1000)
        self.assertEqual(resumed.snapshot()['max_tokens'], 100)

    def test_completed_calls_charge_measured_usage_across_independent_clients(self):
        budget = BudgetClient(Client(), max_tokens=100)
        child = budget.fork(Client(15), allowance=40, usage={'tokens': 0, 'input_tokens': 0})
        list(budget.stream(payload()))
        list(child.stream(payload()))
        self.assertEqual(budget.snapshot()['tokens'], 27)
        self.assertEqual(child.usage['tokens'], 15)
        self.assertEqual(budget.snapshot()['input_tokens'], 40)

    def test_interrupted_call_retains_reservation_on_resume(self):
        class Interrupted(Client):
            def stream(self, payload):
                yield {'message': {'content': 'partial'}}
                raise KeyboardInterrupt
        budget = BudgetClient(Interrupted(), max_tokens=100)
        with self.assertRaises(KeyboardInterrupt):
            list(budget.stream(payload()))
        resumed = BudgetClient(Client(), state=budget.snapshot())
        self.assertEqual(resumed.remaining_tokens, 60)

    def test_cancelled_before_dispatch_does_not_charge_or_call_client(self):
        class Cancelled(Client):
            cancel = threading.Event()

            def stream(self, payload):
                raise AssertionError('Cancelled inference must not be dispatched')
        client = Cancelled()
        client.cancel.set()
        budget = BudgetClient(client, max_tokens=100)
        with self.assertRaises(KeyboardInterrupt):
            list(budget.stream(payload()))
        self.assertEqual(budget.snapshot()['tokens'], 0)
        self.assertEqual(budget.snapshot()['input_tokens'], 0)

    def test_unreported_usage_retains_reservation(self):
        class Unknown(Client):
            def stream(self, payload):
                yield {'done': True}
        budget = BudgetClient(Unknown(), max_tokens=100)
        list(budget.stream(payload()))
        self.assertEqual(budget.snapshot()['tokens'], 40)

    def test_local_allowance_cannot_borrow_another_workers_reservation(self):
        budget = BudgetClient(Client(), max_tokens=100)
        child = budget.fork(Client(), allowance=20, usage={'tokens': 0, 'input_tokens': 0})
        with self.assertRaises(AssistantBudget):
            list(child.stream(payload()))
        self.assertEqual(budget.snapshot()['tokens'], 0)

    def test_concurrent_reservations_cannot_overspend(self):
        entered, release = threading.Event(), threading.Event()
        class Waiting(Client):
            def stream(self, payload):
                entered.set()
                release.wait(2)
                yield {'done': True, 'eval_count': 40}
        budget = BudgetClient(Client(), max_tokens=60)
        first = budget.fork(Waiting(), allowance=60, usage={'tokens': 0, 'input_tokens': 0})
        second = budget.fork(Client(), allowance=60, usage={'tokens': 0, 'input_tokens': 0})
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(lambda: list(first.stream(payload())))
            self.assertTrue(entered.wait(2))
            with self.assertRaises(AssistantBudget):
                list(second.stream(payload()))
            release.set()
            future.result()
        self.assertEqual(budget.snapshot()['tokens'], 40)

    def test_bad_server_usage_is_not_refunded(self):
        budget = BudgetClient(Client(41), max_tokens=100)
        with self.assertRaises(AssistantBudget):
            list(budget.stream(payload()))
        self.assertEqual(budget.snapshot()['tokens'], 40)

    def test_repeated_completion_event_cannot_double_refund(self):
        class Duplicate(Client):
            def stream(self, payload):
                yield {'done': True, 'eval_count': 12}
                yield {'done': True, 'eval_count': 12}
        budget = BudgetClient(Duplicate(), max_tokens=100)
        with self.assertRaises(AssistantBudget):
            list(budget.stream(payload()))
        self.assertEqual(budget.snapshot()['tokens'], 12)
