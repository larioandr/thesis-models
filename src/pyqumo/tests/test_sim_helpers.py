import pytest

from pyqumo.sim.helpers import Queue


# ###########################################################################
# TEST Queue
# ###########################################################################
@pytest.mark.parametrize('capacity', [0, 5])
def test_finite_queue_props(capacity):
    """
    Validate capacity, empty and full properties of a queue with finite cap.
    """
    queue: Queue[int] = Queue(capacity)
    assert queue.capacity == capacity
    assert queue.size == 0
    assert queue.empty

    if capacity == 0:
        assert queue.full

    # Fill half of the queue with values, check it is neither empty nor full:
    half_capacity = capacity // 2
    for i in range(half_capacity):
        queue.push(i)

    assert queue.size == half_capacity
    if half_capacity > 0:
        assert not queue.empty
        assert not queue.full

    # Fill the rest of the queue and make sure it becomes full then:
    for _ in range(half_capacity, capacity):
        queue.push(0)
    assert queue.size == capacity
    assert queue.full


def test_queue_is_fifo():
    """
    Validate push() and pop() works in FIFO order.
    """
    # Put 10 [OK] => 20 [OK] => 30 [LOST]
    queue: Queue[int] = Queue(2)
    assert queue.push(10)
    assert queue.push(20)
    assert not queue.push(30)
    # queue = [10, 20]
    # Pop 10 => { queue = [20] } => push 40 => { queue = [20, 40] }
    assert queue.pop() == 10
    assert queue.push(40)
    # queue = [20, 40]
    # Pop 20 => Pop 40 => Pop [NONE]
    assert queue.pop() == 20
    assert queue.pop() == 40
    assert queue.pop() is None
    assert queue.empty


def test_queue_str():
    """
    Validate __repr__() method of the Queue.
    """
    queue: Queue[int] = Queue(5)
    assert str(queue) == "(Queue: q=[], capacity=5)"
    queue.push(34)
    queue.push(42)
    assert str(queue) == "(Queue: q=[34, 42], capacity=5)"
    queue.push(1)
    queue.push(2)
    queue.push(3)
    assert str(queue) == "(Queue: q=[34, 42, 1, 2, 3], capacity=5)"
