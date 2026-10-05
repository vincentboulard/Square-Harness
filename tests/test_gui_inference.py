"""Real-thread admission tests without contacting an inference server."""
import threading
import time
import unittest

from mathagent.gui.inference import InferencePool


class _Worker:
    def __init__(self, pool, owner, deadline=None):
        self.cancel = threading.Event()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.errors = []

        def run():
            try:
                with pool.slot(owner, self.cancel, deadline=deadline):
                    self.entered.set()
                    if not self.release.wait(5):
                        raise AssertionError('Test did not release the inference worker.')
            except BaseException as exc:
                self.errors.append(exc)
            finally:
                self.finished.set()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def close(self):
        self.cancel.set()
        self.release.set()
        self.thread.join(2)


class InferencePoolTests(unittest.TestCase):
    def setUp(self):
        self.pool = InferencePool()
        self.pool.register('a')

    def worker(self, owner='a', *, deadline=None):
        worker = _Worker(self.pool, owner, deadline)
        self.addCleanup(worker.close)
        return worker

    def entered(self, worker):
        self.assertTrue(worker.entered.wait(2), 'Worker failed to enter a free slot.')
        self.assertEqual(worker.errors, [])

    def finish(self, worker):
        worker.release.set()
        self.assertTrue(worker.finished.wait(2), 'Worker failed to release its slot.')
        self.assertEqual(worker.errors, [])

    def waiters(self, count):
        # Observe admission state to construct deterministic thread schedules,
        # instead of relying on arbitrary sleeps to order competing workers.
        with self.pool._condition:
            self.assertTrue(self.pool._condition.wait_for(
                lambda: len(self.pool._waiters) == count, timeout=2), 'Expected waiters did not queue.')

    def test_lone_task_uses_two_slots_and_a_third_waits(self):
        first, second = self.worker(), self.worker()
        self.entered(first)
        self.entered(second)
        third = self.worker()
        self.waiters(1)
        self.assertFalse(third.entered.is_set())
        self.finish(first)
        self.entered(third)
        self.assertFalse(second.finished.is_set())

    def test_two_registered_tasks_each_get_one_slot(self):
        self.pool.register('b')
        first = self.worker()
        self.entered(first)
        blocked = self.worker()
        self.waiters(1)
        second = self.worker('b')
        self.entered(second)
        self.assertFalse(blocked.entered.is_set())
        self.finish(first)
        self.entered(blocked)
        self.assertFalse(second.finished.is_set())

    def test_new_task_gets_released_slot_before_busy_owners_waiters(self):
        first, second = self.worker(), self.worker()
        self.entered(first)
        self.entered(second)
        blocked = self.worker()
        self.waiters(1)
        self.pool.register('b')
        newcomer = self.worker('b')
        self.waiters(2)
        self.assertFalse(first.finished.is_set())
        self.assertFalse(second.finished.is_set())
        self.finish(first)
        self.entered(newcomer)
        self.assertFalse(blocked.entered.is_set())
        self.finish(second)
        self.entered(blocked)

    def test_fifo_eligible_waiters_prevent_starvation_with_one_slot(self):
        self.pool.set_capacity(1)
        self.pool.register('b')
        active = self.worker()
        self.entered(active)
        first = self.worker('b')
        self.waiters(1)
        second = self.worker()
        self.waiters(2)
        self.finish(active)
        self.entered(first)
        self.assertFalse(second.entered.is_set())
        # Even a new request by the current owner goes behind the waiting task.
        third = self.worker('b')
        self.waiters(2)
        self.finish(first)
        self.entered(second)
        self.assertFalse(third.entered.is_set())
        self.finish(second)
        self.entered(third)

    def test_cancel_while_waiting_removes_waiter_without_releasing_active_slots(self):
        first, second = self.worker(), self.worker()
        self.entered(first)
        self.entered(second)
        canceled = self.worker()
        self.waiters(1)
        canceled.cancel.set()
        self.assertTrue(canceled.finished.wait(1), 'Waiting cancellation was not responsive.')
        self.assertEqual(len(canceled.errors), 1)
        self.assertIsInstance(canceled.errors[0], KeyboardInterrupt)
        self.assertFalse(canceled.entered.is_set())
        self.waiters(0)
        following = self.worker()
        self.waiters(1)
        self.finish(first)
        self.entered(following)
        self.assertFalse(second.finished.is_set())

    def test_exception_and_generator_close_release_the_slot(self):
        self.pool.set_capacity(1)
        with self.assertRaisesRegex(RuntimeError, 'client failed'):
            with self.pool.slot('a', threading.Event()):
                raise RuntimeError('client failed')

        def stream():
            with self.pool.slot('a', threading.Event()):
                yield 'first token'
                yield 'second token'

        source = stream()
        self.addCleanup(source.close)
        self.assertEqual(next(source), 'first token')
        following = self.worker()
        self.waiters(1)
        source.close()
        self.entered(following)

    def test_decreasing_capacity_drains_existing_streams(self):
        first, second = self.worker(), self.worker()
        self.entered(first)
        self.entered(second)
        self.pool.set_capacity(1)
        self.assertEqual(self.pool.capacity, 1)
        following = self.worker()
        self.waiters(1)
        self.finish(first)
        self.assertFalse(following.entered.is_set())
        self.assertFalse(second.finished.is_set())
        self.finish(second)
        self.entered(following)

    def test_increasing_capacity_wakes_waiter(self):
        self.pool.set_capacity(1)
        first = self.worker()
        self.entered(first)
        second = self.worker()
        self.waiters(1)
        self.pool.set_capacity(2)
        self.entered(second)
        self.assertFalse(first.finished.is_set())

    def test_unregister_wakes_waiters_and_restores_lone_owner_allowance(self):
        self.pool.register('b')
        active_a, active_b = self.worker(), self.worker('b')
        self.entered(active_a)
        self.entered(active_b)
        waiting_b = self.worker('b')
        self.waiters(1)
        self.pool.unregister('b')
        self.assertTrue(waiting_b.finished.wait(1))
        self.assertIsInstance(waiting_b.errors[0], KeyboardInterrupt)
        waiting_a = self.worker()
        self.waiters(1)
        self.assertFalse(waiting_a.entered.is_set())
        # The retired owner's active stream keeps consuming capacity.
        self.finish(active_b)
        self.entered(waiting_a)
        self.assertFalse(active_a.finished.is_set())
        self.pool.unregister('b')  # Repeated retirement is harmless.

    def test_repeated_registration_does_not_reduce_owner_allowance(self):
        self.pool.register('a')
        first, second = self.worker(), self.worker()
        self.entered(first)
        self.entered(second)

    def test_pre_canceled_request_does_not_consume_a_slot(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(KeyboardInterrupt):
            with self.pool.slot('a', cancel):
                self.fail('Canceled request entered the pool.')
        first, second = self.worker(), self.worker()
        self.entered(first)
        self.entered(second)

    def test_expired_deadline_rejects_an_otherwise_free_slot(self):
        with self.assertRaisesRegex(TimeoutError, 'time budget exhausted while waiting for inference'):
            with self.pool.slot('a', threading.Event(), deadline=time.monotonic() - 1):
                self.fail('An expired request entered the pool.')
        first, second = self.worker(), self.worker()
        self.entered(first)
        self.entered(second)

    def test_waiting_deadline_expires_without_leaking_a_slot(self):
        self.pool.set_capacity(1)
        active = self.worker()
        self.entered(active)
        waiting = self.worker(deadline=time.monotonic() + 0.08)
        self.waiters(1)
        self.assertTrue(waiting.finished.wait(1), 'Admission did not honor its deadline.')
        self.assertFalse(waiting.entered.is_set())
        self.assertIsInstance(waiting.errors[0], TimeoutError)
        self.assertFalse(active.finished.is_set())
        self.waiters(0)
        following = self.worker(deadline=time.monotonic() + 2)
        self.waiters(1)
        self.finish(active)
        self.entered(following)

    def test_cancellation_takes_precedence_over_expired_deadline(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(KeyboardInterrupt):
            with self.pool.slot('a', cancel, deadline=time.monotonic() - 1):
                self.fail('Canceled request entered the pool.')

    def test_nonfinite_or_invalid_deadline_is_rejected_without_consuming_slots(self):
        for deadline in (float('inf'), float('-inf'), float('nan'), True, 'soon'):
            with self.subTest(deadline=deadline):
                with self.assertRaises(ValueError):
                    with self.pool.slot('a', threading.Event(), deadline=deadline):
                        self.fail('An invalid deadline entered the pool.')
        first, second = self.worker(), self.worker()
        self.entered(first)
        self.entered(second)

    def test_unknown_owner_and_invalid_capacity_are_rejected(self):
        with self.assertRaises(ValueError):
            with self.pool.slot('unknown', threading.Event()):
                self.fail('Unknown owner entered the pool.')
        for value in (0, 9, -1, 1.5, True, '2', None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    InferencePool(value)
                with self.assertRaises(ValueError):
                    self.pool.set_capacity(value)
                self.assertEqual(self.pool.capacity, 2)


if __name__ == '__main__':
    unittest.main()
