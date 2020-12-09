import unittest

import pyons.queues as queues


class TestEmptyHeapQueueFilling(unittest.TestCase):

    def test_queue_push(self):
        queue = queues.HeapQueue(time_getter=lambda t: t[0],
                                 index_getter=lambda t: t[1])
        self.assertTrue(queue.push((0, 0, 'hello')) is not None)
        self.assertTrue(queue.push((5, 1, 'bye')) is not None)
        self.assertTrue(queue.push((3, 2, 'working')) is not None)
        self.assertTrue(queue.push((3, 3, 'dining')) is not None)
        self.assertEqual(len(queue), 4)
        self.assertFalse(queue.empty)
        self.assertEqual(queue.pop()[-1], 'hello')
        self.assertEqual(queue.pop()[-1], 'working')
        self.assertEqual(queue.pop()[-1], 'dining')
        self.assertEqual(queue.pop()[-1], 'bye')
        self.assertEqual(len(queue), 0)
        self.assertTrue(queue.empty)


class TestHeapQueueOperations(unittest.TestCase):
    def setUp(self):
        self.queue = queues.HeapQueue(time_getter=lambda t: t[0],
                                      index_getter=lambda t: t[1])
        self.hello_id = self.queue.push((0, 0, 'hello'))
        self.bye_id = self.queue.push((5, 1, 'bye'))
        self.working_id = self.queue.push((3, 2, 'working'))
        self.dining_id = self.queue.push((3, 3, 'dining'))

    def tearDown(self):
        self.queue.clear()

    def test_queue_remove_by_index(self):
        self.assertEqual(len(self.queue), 4)

        #
        # Try to remove by the index that doesn't
        # exist => length shouldn't change
        #
        self.queue.remove(index=max([self.hello_id, self.bye_id,
                                     self.working_id, self.dining_id]) + 1)
        self.assertEqual(len(self.queue), 4)
        #
        # Remove 'bye' event
        #
        self.queue.remove(index=self.bye_id)
        self.assertEqual(len(self.queue), 3)
        #
        # Remove 'dining' event. The second attempt shouldn't
        # change the queue length
        #
        self.queue.remove(index=self.dining_id)
        self.assertEqual(len(self.queue), 2)
        self.queue.remove(index=self.dining_id)
        self.assertEqual(len(self.queue), 2)
        #
        # Making sure that 'hello' and 'working' are still there
        #
        self.assertEqual(self.queue.pop()[-1], 'hello')
        self.assertEqual(self.queue.pop()[-1], 'working')
        #
        # Filling the items back and removing them till the queue end
        #
        self.hello_id = self.queue.push((0, 4, 'hello'))
        self.working_id = self.queue.push((1, 5, 'working'))
        self.queue.remove(index=self.working_id)
        self.assertEqual(len(self.queue), 1)
        self.queue.remove(index=self.hello_id)
        self.assertEqual(len(self.queue), 0)

    def test_queue_remove_by_predicate(self):
        self.assertEqual(len(self.queue), 4)
        #
        # Remove all the events ending with 'ing'
        # ('working' and 'dining' here).
        # The second call is for making sure that the second
        # attempt doesn't change the queue length.
        #
        self.queue.remove(predicate=lambda event: event[-1][-3:] == 'ing')
        self.assertEqual(len(self.queue), 2)
        self.queue.remove(predicate=lambda event: event[-1][-3:] == 'ing')
        self.assertEqual(len(self.queue), 2)
        #
        # Making sure that 'hello' and 'bye' are still there
        #
        self.assertEqual(self.queue.pop()[-1], 'hello')
        self.assertEqual(self.queue.pop()[-1], 'bye')
        self.assertTrue(self.queue.empty)
        #
        # Putting 'hello' and 'bye' back and removing them using predicates
        #
        self.queue.push((0, 4, 'hello'))
        self.queue.push((0, 5, 'bye'))
        self.queue.remove(predicate=lambda event: True)
        self.assertTrue(self.queue.empty)

    def test_queue_clear(self):
        self.assertEqual(len(self.queue), 4)
        self.queue.clear()
        self.assertTrue(self.queue.empty)

    def test_queue_events_property(self):
        words = [word for t, i, word in self.queue.items]
        self.assertEqual(words, ['hello', 'working', 'dining', 'bye'])
