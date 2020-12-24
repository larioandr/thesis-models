import pytest
from numpy import argmin, cumsum, inf, asarray
from pydesim import Model, simulate, Statistic, Intervals
from unittest.mock import Mock, patch, ANY
from pycsmaca.simulations.modules.app_layer import RandomSource, AppData, \
    Sink, ControlledSource


class DummyModel(Model):
    """We use this `DummyModel` when we need a full-functioning model.
    """
    def __init__(self, sim, name):
        super().__init__(sim)
        self.name = name

    def __str__(self):
        return self.name


#############################################################################
# TEST RandomSource MODULE
#############################################################################

# noinspection PyProtectedMember
def test_random_source_generates_packets():
    """In this test we check that `RandomSource` properly generates `AppData`.
    """
    # First, we create the `RandomSource` module, validate it is
    # inherited from `pydesim.Module` and check that upon construction source
    # scheduled the next packet arrival as specified by `interval` parameter:
    sim = Mock()
    sim.stime = 0
    source = RandomSource(sim, data_size=Mock(return_value=42),
                          interval=Mock(side_effect=(74, 21)), source_id=34,
                          dest_addr=13)
    assert isinstance(source, Model)
    sim.schedule.assert_called_with(74, source._generate)

    # Define a mock for NetworkLayer module and establish a connection:
    network_service_mock = Mock()
    source.connections['network'] = network_service_mock

    # Now we call method `_generate()` method and make sure that it sends a
    # packet via the 'network' connection.
    # Exactly it means that the connected module `handle_message(packet)`
    # method is called using `sim.schedule`, which is expected to be called
    # from within `source.connections['network']` connection.
    with patch('pycsmaca.simulations.modules.app_layer.AppData') as AppDataMock:
        _spec = dict(dest_addr=13, size=42, source_id=34, created_at=0)
        _packet = Mock(**_spec)
        AppDataMock.return_value = _packet

        source._generate()

        AppDataMock.assert_called_with(**_spec)

        rev_conn = source.connections['network'].reverse
        sim.schedule.assert_any_call(
            0, network_service_mock.handle_message, args=(_packet,),
            kwargs={'sender': source, 'connection': rev_conn}
        )

    # Finally, we make sure that after the _generate() call another event
    # was scheduled:
    sim.schedule.assert_any_call(21, source._generate)


# noinspection PyProtectedMember
def test_random_source_can_use_constant_distributions():
    """Validate that numeric constants can be used instead of distributions.
    """
    sim = Mock()
    sim.stime = 0
    source = RandomSource(
        sim, data_size=123, interval=34, source_id=0, dest_addr=1)

    network_service_mock = Mock()
    source.connections['network'] = network_service_mock
    sim.schedule.assert_called_with(34, source._generate)

    with patch('pycsmaca.simulations.modules.app_layer.AppData') as AppDataMock:
        _spec = dict(dest_addr=1, size=123, source_id=0, created_at=0)
        _packet = Mock(**_spec)
        AppDataMock.return_value = _packet

        source._generate()
        AppDataMock.assert_called_with(**_spec)

    sim.schedule.assert_any_call(34, source._generate)


# noinspection PyProtectedMember
def test_random_source_can_use_finite_intervals_distributions():
    """Validate that `RandomSource` will stop when intervals is finite tuple.
    """
    sim = Mock()
    sim.stime = 0
    source = RandomSource(
        sim, data_size=123, interval=(34, 42,), source_id=0, dest_addr=1)

    network_service_mock = Mock()
    source.connections['network'] = network_service_mock
    rev_conn = source.connections['network'].reverse

    sim.schedule.assert_called_with(34, source._generate)
    sim.schedule.reset_mock()

    source._generate()
    sim.schedule.assert_any_call(42, source._generate)
    sim.schedule.reset_mock()

    source._generate()
    sim.schedule.assert_called_once_with(
        0, network_service_mock.handle_message, args=(ANY,),
        kwargs={'sender': source, 'connection': rev_conn}
    )


# noinspection PyProtectedMember
def test_random_source_can_use_finite_data_size_distributions():
    """Validate that `RandomSource` will stop when data size is finite tuple.
    """
    sim = Mock()
    sim.stime = 0
    source = RandomSource(
        sim, data_size=(10, 20), interval=100, source_id=0, dest_addr=1)

    network_service_mock = Mock()
    source.connections['network'] = network_service_mock

    with patch('pycsmaca.simulations.modules.app_layer.AppData') as AppDataMock:
        source._generate()
        AppDataMock.assert_called_with(dest_addr=1, source_id=0, size=10,
                                       created_at=0)
        AppDataMock.reset_mock()

        source._generate()
        AppDataMock.assert_called_with(dest_addr=1, source_id=0, size=20,
                                       created_at=0)
        AppDataMock.reset_mock()

        sim.schedule.reset_mock()
        source._generate()
        AppDataMock.assert_not_called()
        sim.schedule.assert_not_called()


# noinspection PyProtectedMember
def test_random_source_provides_statistics():
    """Validate that `RandomSource` provides statistics.
    """
    intervals = (10, 12, 15, 17)
    data_size = (123, 453, 245, 321)

    class TestModel(Model):
        def __init__(self, sim):
            super().__init__(sim)
            self.source = RandomSource(
                sim, source_id=34, dest_addr=13,
                data_size=Mock(side_effect=data_size),
                interval=Mock(side_effect=(intervals + (1000,))),
            )
            self.network = DummyModel(sim, 'Network')
            self.source.connections['network'] = self.network

    ret = simulate(TestModel, stime_limit=sum(intervals))

    assert ret.data.source.arrival_intervals.as_tuple() == intervals
    assert ret.data.source.data_size_stat.as_tuple() == data_size

    # Also check that we can not replace statistics:
    with pytest.raises(AttributeError):
        from pydesim import Intervals
        ret.data.source.arrival_intervals = Intervals()
    with pytest.raises(AttributeError):
        from pydesim import Statistic
        ret.data.source.data_size_stat = Statistic()

    # Check that source records the number of packets being sent:
    assert ret.data.source.num_packets_sent == 4


#############################################################################
# TEST AppData PACKETS
#############################################################################
# noinspection PyPropertyAccess
def test_app_data_is_immutable():
    app_data = AppData(dest_addr=5, size=20, source_id=13, created_at=123)
    assert app_data.destination_address == 5
    assert app_data.size == 20
    assert app_data.source_id == 13
    assert app_data.created_at == 123
    with pytest.raises(AttributeError):
        app_data.destination_address = 11
    with pytest.raises(AttributeError):
        app_data.size = 21
    with pytest.raises(AttributeError):
        app_data.source_id = 26
    with pytest.raises(AttributeError):
        app_data.created_at = 234


def test_app_data_supports_default_values():
    app_data = AppData()
    assert app_data.destination_address == 0
    assert app_data.size == 0
    assert app_data.source_id == 0
    assert app_data.created_at == 0


def test_app_data_provides_str():
    app_data = AppData(dest_addr=1, size=250, source_id=2, created_at=10)
    assert str(app_data) == 'AppData{sid=2,dst=1,size=250,ct=10}'


#############################################################################
# TEST Sink MODULE
#############################################################################
def test_sink_module_records_statistics():
    """Validate that `Sink` module records various statistics.

    In this test we check that `Sink` module records end-to-end delays per
    SourceID, inter-arrival intervals and packet sizes (last two statistics
    are recorded without splitting by SourceID). It also records the number of
    received packets.

    To do this, we define a sample simulation with dummy network layer, two
    sources and one sink. Each source generates three messages within given
    intervals and with given sizes.
    """
    delays = [(5, 2, 234), (10, 3, 135)]
    intervals = [(10, 10, 200), (20, 400, 80)]
    sizes = [(1234, 1625, 1452), (2534, 2124, 2664)]
    sids = [23, 14]

    class DummyNetwork(Model):
        def __init__(self, sim):
            super().__init__(sim)

        def handle_message(self, app_data, **kwargs):
            self.connections['sink'].send(app_data)

    class TestModel(Model):
        def __init__(self, sim):
            super().__init__(sim)
            self.sink = Sink(sim)
            for ds, inter, sid, delay in zip(sizes, intervals, sids, delays):
                network = DummyNetwork(sim)
                source = RandomSource(sim, ds, inter, sid, 5)
                source.connections['network'] = network
                network.connections['sink'] = self.sink
                network.connections['sink'].delay = Mock(side_effect=delay)

    ret = simulate(TestModel)
    sink = ret.data.sink

    assert isinstance(sink.source_delays[sids[0]], Statistic)
    assert sink.source_delays[sids[0]].as_tuple() == delays[0]
    assert sink.source_delays[sids[1]].as_tuple() == delays[1]

    # To check `Sink` stores arrival intervals and packet sizes, we estimate
    # expected arrival times (received_at) and along with it order packet
    # sizes by their arrival time:
    indices = [0, 0]
    gen_times = [cumsum(ints) for ints in intervals]
    n = len(intervals[0])
    received_at, received_sizes = [], []
    while any(i < n for i in indices):
        next_arrivals = [
            (gen_times[i][j] + delays[i][j]) if j < n else inf
            for i, j in enumerate(indices)
        ]
        i = int(argmin(next_arrivals))
        received_at.append(next_arrivals[i])
        received_sizes.append(sizes[i][indices[i]])
        indices[i] += 1

    # Since received_at stores arrival timestamps, compute arrival intervals:
    v = asarray(received_at)
    arrival_intervals = [v[0]] + list(v[1:] - v[:-1])

    # Check that recorded data match the expected:
    assert isinstance(sink.arrival_intervals, Intervals)
    assert sink.arrival_intervals.as_tuple() == tuple(arrival_intervals)
    assert isinstance(sink.data_size_stat, Statistic)
    assert sink.data_size_stat.as_tuple() == tuple(received_sizes)
    assert sink.num_packets_received == len(received_sizes)

    # Check that statistics can not be overwritten:
    with pytest.raises(AttributeError):
        sink.arrival_intervals = Intervals()
    with pytest.raises(AttributeError):
        sink.data_size_stat = Statistic()
    with pytest.raises(TypeError):
        sink.source_delays[sids[0]] = Statistic()


def test_sink_model_implements_str():
    sim = Mock()
    sink = Sink(sim)
    assert str(sink) == 'Sink'


#############################################################################
# TEST ControlledSource MODULE
#############################################################################

# noinspection PyProtectedMember
def test_controlled_source_generates_packets():
    """In this test we check that `ControlledSource` generates `AppData`.
    """
    # First, we create the `ControlledSource` module, validate it is
    # inherited from `pydesim.Module` and check that upon construction nothing
    # being scheduled:
    sim = Mock()
    sim.stime = 0
    source = ControlledSource(
        sim, data_size=Mock(return_value=42), source_id=34, dest_addr=13)
    assert isinstance(source, Model)
    sim.schedule.assert_not_called()

    # Define a mock for NetworkLayer module and establish a connection:
    network_service_mock = Mock()
    source.connections['network'] = network_service_mock

    # Now we call method `get_next()` method and make sure that it sends a
    # packet via the 'network' connection.
    # Exactly it means that the connected module `handle_message(packet)`
    # method is called using `sim.schedule`, which is expected to be called
    # from within `source.connections['network']` connection.
    # We also make sure that was the only call, no next arrival scheduled.
    with patch('pycsmaca.simulations.modules.app_layer.AppData') as AppDataMock:
        _spec = dict(dest_addr=13, size=42, source_id=34, created_at=0)
        _packet = Mock(**_spec)
        AppDataMock.return_value = _packet

        source.get_next()

        AppDataMock.assert_called_with(**_spec)

        rev_conn = source.connections['network'].reverse
        sim.schedule.assert_called_once_with(
            0, network_service_mock.handle_message, args=(_packet,),
            kwargs={'sender': source, 'connection': rev_conn}
        )


# noinspection PyProtectedMember
def test_controlled_source_can_use_constant_size_distribution():
    """Validate that numeric constant can be used instead of size distribution.
    """
    sim = Mock()
    sim.stime = 0
    source = ControlledSource(sim, data_size=123, source_id=0, dest_addr=1)

    network_service_mock = Mock()
    source.connections['network'] = network_service_mock

    with patch('pycsmaca.simulations.modules.app_layer.AppData') as AppDataMock:
        _spec = dict(dest_addr=1, size=123, source_id=0, created_at=0)
        _packet = Mock(**_spec)
        AppDataMock.return_value = _packet

        source.get_next()

        AppDataMock.assert_called_with(**_spec)


# noinspection PyProtectedMember
def test_controlled_source_can_use_finite_data_size_distributions():
    """Validate `ControlledSource` will stop if data size is a finite tuple.
    """
    sim = Mock()
    sim.stime = 0
    source = ControlledSource(sim, data_size=(10, 20), source_id=0, dest_addr=1)

    network_service_mock = Mock()
    source.connections['network'] = network_service_mock

    with patch('pycsmaca.simulations.modules.app_layer.AppData') as AppDataMock:
        source.get_next()
        AppDataMock.assert_called_with(
            dest_addr=1, source_id=0, size=10, created_at=0)
        AppDataMock.reset_mock()

        sim.stime = 5.2
        source.get_next()
        AppDataMock.assert_called_with(
            dest_addr=1, source_id=0, size=20, created_at=5.2)
        AppDataMock.reset_mock()

        sim.schedule.reset_mock()
        source.get_next()
        AppDataMock.assert_not_called()
        sim.schedule.assert_not_called()


# noinspection PyProtectedMember
def test_controlled_source_provides_statistics():
    """Validate that `ControlledSource` provides statistics.
    """
    intervals = (10, 12, 15, 17)
    data_size = (123, 453, 245, 321)

    class SourceController(Model):
        def __init__(self, sim, src):
            super().__init__(sim)
            self.iterator = iter(intervals)
            self.src = src
            self.sim.schedule(next(self.iterator), self.handle_timeout)

        def handle_timeout(self):
            self.src.get_next()
            try:
                interval = next(self.iterator)
            except StopIteration:
                pass
            else:
                self.sim.schedule(interval, self.handle_timeout)

    class TestModel(Model):
        def __init__(self, sim):
            super().__init__(sim)
            self.source = ControlledSource(
                sim, source_id=34, dest_addr=13,
                data_size=Mock(side_effect=data_size),
            )
            self.network = DummyModel(sim, 'Network')
            self.source.connections['network'] = self.network
            self.controller = SourceController(sim, self.source)

    ret = simulate(TestModel, stime_limit=sum(intervals))

    assert ret.data.source.data_size_stat.as_tuple() == data_size
    assert ret.data.source.arrival_intervals.as_tuple() == intervals

    # Also check that we can not replace statistics:
    with pytest.raises(AttributeError):
        from pydesim import Intervals
        ret.data.source.arrival_intervals = Intervals()
    with pytest.raises(AttributeError):
        from pydesim import Statistic
        ret.data.source.data_size_stat = Statistic()

    # Check that source records the number of packets being sent:
    assert ret.data.source.num_packets_sent == 4
