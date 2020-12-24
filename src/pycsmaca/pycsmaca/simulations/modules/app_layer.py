from pydesim import Model, Intervals, Statistic

from pycsmaca.utilities import ReadOnlyDict


class AppData:
    def __init__(self, dest_addr=0, size=0, source_id=0, created_at=0):
        self.__dest_addr = dest_addr
        self.__size = size
        self.__source_id = source_id
        self.__created_at = created_at

    @property
    def destination_address(self):
        return self.__dest_addr

    @property
    def size(self):
        return self.__size

    @property
    def source_id(self):
        return self.__source_id

    @property
    def created_at(self):
        return self.__created_at

    def __str__(self):
        fields = ','.join([
            f'sid={self.source_id}', f'dst={self.destination_address}',
            f'size={self.size}', f'ct={self.created_at}'
        ])
        return f'AppData{{{fields}}}'


class _SourceBase(Model):
    def __init__(self, sim, data_size, source_id, dest_addr):
        """Constructor.

        :param sim: `pydesim.Simulator` object;
        :param data_size: callable without arguments, iterable or constant;
            represents application data size distribution;
        :param source_id: this source ID (more like IP address, not MAC)
        :param dest_addr: destination MAC address.
        """
        super().__init__(sim)
        self.__data_size = data_size
        self.__source_id = source_id
        self.__dest_addr = dest_addr

        # Attempt to build iterators for data size and intervals:
        try:
            self.__data_size_iter = iter(self.__data_size)
        except TypeError:
            self.__data_size_iter = None

        # Statistics:
        self.__arrival_intervals = Intervals()
        self.__data_size_stat = Statistic()
        self.__num_packets_sent = 0

    @property
    def arrival_intervals(self):
        return self.__arrival_intervals

    @property
    def data_size_stat(self):
        return self.__data_size_stat

    @property
    def data_size(self):
        return self.__data_size

    @property
    def source_id(self):
        return self.__source_id

    @property
    def dest_addr(self):
        return self.__dest_addr

    @property
    def num_packets_sent(self):
        return self.__num_packets_sent

    def _generate(self):
        try:
            data_size = self.__get_next_size()
        except StopIteration:
            return False # do nothing if stop iteration fired
        else:
            app_data = AppData(
                dest_addr=self.dest_addr, size=data_size,
                source_id=self.source_id, created_at=self.sim.stime
            )
            self.connections['network'].send(app_data)
            # Recording statistics:
            self.arrival_intervals.record(self.sim.stime)
            self.data_size_stat.append(data_size)
            self.__num_packets_sent += 1
            self.sim.logger.debug(f'generated new packet {app_data}', src=self)
            return True

    def __get_next_size(self):
        if self.__data_size_iter:
            return next(self.__data_size_iter)
        try:
            return self.data_size()
        except TypeError:
            return self.data_size

    def __str__(self):
        prefix = f'{self.parent}.' if self.parent else ''
        return f'{prefix}Source({self.source_id})'


class RandomSource(_SourceBase):
    """This module provides data source with independent intervals and sizes.

    `RandomSource` generates `AppData` packets with a given bit size
    distribution. Inter-arrival intervals are also randomly distributed,
    and this distribution is independent from data size distribution.

    Note: distributions are passed to the constructor as callable objects,
    but they also can be specified with constants.

    Source directs its packets to network layer. Packets have a given
    destination address. Source is specified with its SourceID.

    Provided statistics:
    - `arrival_intervals`: inter-arrival intervals;
    - `data_size_stat`: generated data sizes statistics.

    Events:
    - timeout: fired when inter-arrival interval is reached.

    Connections:
    - 'network': connected network layer module; should implement
        `handle_message(app_data)` method.
    """
    def __init__(self, sim, data_size, interval, source_id, dest_addr):
        """Create `RandomSource` module.

        :param sim: `pydesim.Simulator` object;
        :param data_size: callable without arguments, iterable or constant;
            represents application data size distribution;
        :param interval: callable without arguments, iterable or constant;
            represents inter-arrival intervals distribution;
        :param source_id: this source ID (more like IP address, not MAC)
        :param dest_addr: destination MAC address.
        """
        super().__init__(sim, data_size, source_id, dest_addr)
        self.__interval = interval

        # Attempt to build iterators for data size and intervals:
        try:
            self.__interval_iter = iter(self.__interval)
        except TypeError:
            self.__interval_iter = None

        # Initialize:
        self._schedule_next_arrival()

    @property
    def interval(self):
        return self.__interval

    def _generate(self):
        if super()._generate():
            self._schedule_next_arrival()
            return True
        return False

    def _get_next_interval(self):
        if self.__interval_iter is not None:
            return next(self.__interval_iter)
        try:
            return self.interval()
        except TypeError:
            return self.interval

    def _schedule_next_arrival(self):
        try:
            self.sim.schedule(self._get_next_interval(), self._generate)
        except StopIteration:
            pass


class ControlledSource(_SourceBase):
    """This module provides a data source generating packets on request.

    `ControlledSource` generates `AppData` packets with a given bit size
    distribution. Generation takes place when `get_next()` method is called.

    Note: size distribution are passed to the constructor as callable objects,
    but they also can be specified with constants or iterables.

    Source directs its packets to network layer. Packets have a given
    destination address. Source is specified with its SourceID.

    Provided statistics:
    - `arrival_intervals`: inter-arrival intervals;
    - `data_size_stat`: generated data sizes statistics.

    Connections:
    - 'network': connected network layer module; should implement
        `handle_message(app_data)` method.
    """
    def __init__(self, sim, data_size, source_id, dest_addr):
        """Create `ControlledSource` module.

        :param sim: `pydesim.Simulator` object;
        :param data_size: callable without arguments, iterable or constant;
            represents application data size distribution;
        :param source_id: this source ID (more like IP address, not MAC)
        :param dest_addr: destination MAC address.
        """
        super().__init__(sim, data_size, source_id, dest_addr)

    def get_next(self):
        self._generate()


class Sink(Model):
    """Accepts `AppData` sink, records end-to-end delays and other statistics.

    This module is expected to be placed on top of network layer. It accepts
    any `AppData` and records end-to-end delays per source ID. Besides that,
    it records inter-arrival times for all received packets, as well as
    their sizes, without differentiating them by source ID.

    Statistics:
    - `source_delays`: `ReadOnlyDict`, storing `Statistic` objects as values
        and source IDs as keys;
    - `arrival_intervals`: `Interval` object, stores inter-arrival intervals;
    - `data_size_stat`: `Statistic`, stores received payload sizes.

    Connections: doesn't specify connections, however, expected to be bound
        to network layer.

    Handlers:
    - `handle_message(app_data)`: anything received is treated as `AppData`
        from network layer; connection names or modules not analyzed.
    """
    def __init__(self, sim):
        super().__init__(sim)
        self.__source_delays_data = {}
        self.__source_delays = ReadOnlyDict(self.__source_delays_data)
        self.__arrival_intervals = Intervals()
        self.__data_size_stat = Statistic()
        self.__num_packets_received = 0

    @property
    def arrival_intervals(self):
        return self.__arrival_intervals

    @property
    def data_size_stat(self):
        return self.__data_size_stat

    @property
    def source_delays(self):
        return self.__source_delays

    @property
    def num_packets_received(self):
        return self.__num_packets_received

    def handle_message(self, app_data, sender=None, connection=None):
        sid = app_data.source_id
        if sid not in self.source_delays:
            self.__source_delays_data[sid] = Statistic()
        self.source_delays[sid].append(self.sim.stime - app_data.created_at)
        self.arrival_intervals.record(self.sim.stime)
        self.data_size_stat.append(app_data.size)
        self.__num_packets_received += 1
        self.sim.logger.debug(f'received {app_data}', src=self)

    def __str__(self):
        prefix = f'{self.parent}.' if self.parent else ''
        return f'{prefix}Sink'
