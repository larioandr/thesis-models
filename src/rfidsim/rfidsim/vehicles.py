import numpy as np

import pyons
from pyons import Entity
from . import journal


class Vehicle(Entity):
    def __init__(self):
        super().__init__()
        self.vehicle_id = None
        self.front_tag = None
        self.back_tag = None
        self.lifetime = 2.0
        self.speed = 20.0
        self.length = 4.0
        self.lane = None
        self.direction = None

        self._created_at = None

    @Entity.initializer(stage=(0, 'vehicle initialization'))
    def _initialize_vehicle(self):
        vrec = journal.VehicleInfoRecord()
        vrec.vehicle_id = self.vehicle_id
        vrec.created_at = pyons.time()
        vrec.destroyed_at = None
        vrec.direction = self.direction
        vrec.lane = self.lane
        vrec.speed = self.speed
        vrec.n_read = 0
        journal.Journal().write_vehicle_created(vrec)

        self._created_at = pyons.time()
        if self.front_tag is not None:
            self.front_tag.vehicle = self
            vrec.front_tag_epc = self.front_tag.epc
            vrec.front_tag_tid = self.front_tag.tid
            pyons.get_model().add_tag(self.front_tag)
        if self.back_tag is not None:
            self.back_tag.vehicle = self
            vrec.back_tag_epc = self.back_tag.epc
            vrec.back_tag_tid = self.back_tag.tid
            pyons.get_model().add_tag(self.back_tag)
        pyons.debug(
            f"created vehicle {self.vehicle_id} with "
            f"front tag epc={self.front_tag.epc if self.front_tag else '-'}, "
            f"back tag epc={self.back_tag.epc if self.back_tag else '-'}")

    @Entity.death_condition()
    def _vehicle_suicide(self):
        return pyons.time() - self._created_at > self.lifetime

    @Entity.finalizer(stage=None)
    def _vehicle_finalizer(self):
        t = pyons.time()
        journal.Journal().write_vehicle_destroyed(self.vehicle_id, t)
        if self.front_tag is not None:
            pyons.get_model().remove_tag(self.front_tag)
        if self.back_tag is not None:
            pyons.get_model().remove_tag(self.back_tag)
        pyons.get_model().remove_vehicle(self)
        pyons.debug(f"removed vehicle {self.vehicle_id}")


