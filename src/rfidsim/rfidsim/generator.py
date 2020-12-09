import time as systime

import pyons
from pyons import Entity
from factory import Factory


class Generator(Entity):
    def __init__(self, model_descriptor):
        super().__init__()
        self.params = model_descriptor
        self.n_generated = None
        self.t_started = None

    @Entity.initializer((0.1, 'generator initialization'))
    def _initializer(self):
        self.n_generated = 0
        self.t_started = systime.time()
        for lane in range(0, self.params.lanes_number):
            pyons.create_timeout(self.params.vehicle_generation_interval(),
                                 ('generate vehicle', lane))

    @Entity.stop_condition()
    def _check_generated_enough(self):
        return self.params.max_vehicles_num is not None and \
               self.n_generated > self.params.max_vehicles_num

    @Entity.eventhandler(lambda ev, src: (isinstance(ev, tuple) and
                                          ev[0] == 'generate vehicle'))
    def _handle_generate_vehicle(self, event, source):
        msg, lane = event
        assert source is self and msg == 'generate vehicle'
        model = pyons.get_model()
        vehicle = Factory().build_vehicle(lane, model.channel)
        model.add_vehicle(vehicle)
        pyons.create_timeout(self.params.vehicle_generation_interval(),
                             ('generate vehicle', lane))
        self.n_generated += 1
        if self.n_generated % 10 == 0:
            print("{:4.1f} [GENERATOR] created {} vehicles".format(
                systime.time() - self.t_started, self.n_generated))
