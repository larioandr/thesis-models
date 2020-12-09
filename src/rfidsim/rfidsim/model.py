import pyons


class Model(object):
    def __init__(self, descriptor):
        super().__init__()
        self.position_update_interval = \
            descriptor.vehicle_position_update_interval
        self._channel = None
        self._tags = {}
        self._reader = None
        self._vehicles = []
        self._generator = None
        self.params = descriptor

    @property
    def tags(self):
        return self._tags.values()

    @property
    def vehicles(self):
        return self._vehicles

    @property
    def channel(self):
        return self._channel

    @property
    def reader(self):
        return self._reader

    @property
    def generator(self):
        return self._generator

    @channel.setter
    def channel(self, channel):
        if channel is not self._channel:
            if self._channel is not None:
                pyons.remove_entity(self._channel)
            self._channel = channel
            if channel is not None:
                pyons.add_entity(channel)

    @reader.setter
    def reader(self, reader):
        if reader is not self._reader:
            if self._reader is not None:
                pyons.remove_entity(self._reader)
            self._reader = reader
            if reader is not None:
                pyons.add_entity(reader)

    @generator.setter
    def generator(self, generator):
        if generator is not self._generator:
            if self._generator is not None:
                pyons.remove_entity(self._generator)
            self._generator = generator
            if generator is not None:
                pyons.add_entity(generator)

    def add_tag(self, tag):
        assert tag.epc not in self.tags
        self._tags[tag.epc] = tag
        pyons.add_entity(tag)

    def remove_tag(self, tag):
        if tag.epc in self._tags:
            del self._tags[tag.epc]
            pyons.remove_entity(tag)

    def get_tag(self, epc):
        if epc in self._tags:
            return self._tags[epc]
        raise RuntimeError("tag with epc={} not found", epc)

    def add_vehicle(self, vehicle):
        self.vehicles.append(vehicle)
        pyons.add_entity(vehicle)

    def remove_vehicle(self, vehicle):
        if vehicle in self.vehicles:
            self.vehicles.remove(vehicle)
            pyons.remove_entity(vehicle)
