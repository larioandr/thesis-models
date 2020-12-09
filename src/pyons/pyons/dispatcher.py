from enum import Enum
import itertools

from . import entities as en
from . import errors
from .base import Singleton
from . import queues


#######################################################################
# DISPATCHER
#######################################################################

class Dispatcher(object, metaclass=Singleton):
    class State(Enum):
        READY = 0
        INITIALIZING = 1
        RUNNING = 2
        FINISHING = 3
        FINISHED = 4

    class Envelope(object):
        def __init__(self, index, event, source=None, target=None,
                     handler=None, fire_time=None):
            self.time = fire_time
            self.index = index
            self.event = event
            self.source = source
            self.target = target
            self.handler = handler

    def __init__(self):
        self._state = Dispatcher.State.READY
        self._time = 0
        self._next_event_index = itertools.count()
        self._delayed_queue = queues.HeapQueue()
        self._immediate_queue = []
        self._entities_registry = en.EntitiesRegistry()
        self._entities_cache = []  # (entity, add|remove)
        self._aborted = False
        self._stopped = False
        self._num_events_served = 0
        self.model = None  # this could be set to access model

        self.log_queue = False
        self.log_entities = False
        self.logging_interval = 10

        self.immediate_queue_size = [0]
        self.delayed_queue_size = [0]
        self.total_queue_size = [0]
        self.entities_per_status_num = {
            en.EntitiesRegistry.Stage.NEW: [0],
            en.EntitiesRegistry.Stage.INITIALIZED: [0],
            en.EntitiesRegistry.Stage.FINISHED: [0]
        }
        self.total_entities_num = [0]

    @property
    def time(self):
        return self._time

    @property
    def state(self):
        return self._state
    
    @property
    def events(self):
        return self._immediate_queue + self._delayed_queue.items

    @property
    def num_events_served(self):
        return self._num_events_served

    def schedule(self, event, fire_time=None, target=None, handler=None):
        envelope = Dispatcher.Envelope(
            next(self._next_event_index), event,
            fire_time=(fire_time if fire_time is not None else self.time),
            source=en.CallStack().entity,
            target=target, handler=handler)
        if fire_time is None:
            self._immediate_queue.append(envelope)
        else:
            self._delayed_queue.push(envelope)
        return envelope.index

    def cancel(self, event_id=None, entity=None):
        assert isinstance(self._delayed_queue, queues.Queue)

        def entities_match(envelope):
            return envelope.source is entity or envelope.target is entity

        self._delayed_queue.remove(
            index=event_id, predicate=(entities_match if entity is not None
                                       else None))

        if entity is None:
            self._immediate_queue = list(filter(
                lambda envelope: envelope.index != event_id,
                self._immediate_queue))
        else:
            if event_id is None:
                self._immediate_queue = list(filter(
                    lambda envelope: not entities_match(envelope),
                    self._immediate_queue))
            else:
                self._immediate_queue = list(filter(
                    lambda envelope: (not entities_match(envelope)
                                      and envelope.index != event_id),
                    self._immediate_queue))

    def attach(self, entity):
        if self._state is Dispatcher.State.READY:
            self._entities_registry.add(entity)
        elif self._state is Dispatcher.State.INITIALIZING:
            self._entities_cache.append((entity, 'add', None))
        elif self._state is Dispatcher.State.RUNNING:
            self._entities_registry.add(entity)
            en.initialize(self._entities_registry, entity=entity)
        else:
            pass  # do nothing since dispatcher is finishing

    def detach(self, entity, abort=False):
        if self._state is Dispatcher.State.READY:
            self._entities_registry.remove(entity)
        elif self._state is Dispatcher.State.INITIALIZING:
            self._entities_cache = [
                (e, op, ab) for (e, op, ab) in self._entities_cache
                if e is not entity and op != 'add']
            self._entities_cache.append((entity, 'remove', abort))
        elif self._state is Dispatcher.State.RUNNING:
            self.cancel(entity=entity)
            if not abort:
                en.finalize(self._entities_registry, entity=entity)
            self._entities_registry.remove(entity)
        elif self._state is Dispatcher.State.FINISHING:
            self._entities_cache.append((entity, 'remove', abort))
        else:
            self._entities_registry.remove(entity)

    def start(self, max_events=None, queue=None):
        if self._state != Dispatcher.State.READY:
            return

        self._delayed_queue = queue if queue is not None else \
            queues.HeapQueue(time_getter=lambda e: e.time,
                             index_getter=lambda e: e.index)

        # 1) Initialization
        fine("initializing", self.__class__.__name__)
        self._state = Dispatcher.State.INITIALIZING
        en.initialize(entities_registry=self._entities_registry,
                      static_registry=en.StaticRegistry())

        # ..) completing initialization by adding and removing
        #     cached entities
        self._state = Dispatcher.State.RUNNING
        for entity, op, abort in self._entities_cache:
            if op == 'add':
                self.attach(entity)
            elif op == 'remove':
                self.detach(entity, abort)
        self._entities_cache.clear()

        fine("simulation started", self.__class__.__name__)
        # 2) running main event loop
        while not self._stopped:

            # if self._num_events_served % 1e4 == 0:
            #     print("DISPATCHER: t={:09.7f}s, served {:08} events".format(
            #         self.time, self._num_events_served))

            # 2.1) looking for entities going to die and killing them
            death_list = en.check_death_conditions(self._entities_registry)
            for cond, entity in death_list:
                fine("[-x-] death condition={}, killing entity={}"
                      "".format(cond.get_name(), str(entity)),
                      self.__class__.__name__)
                self.cancel(entity=entity)
                en.finalize(self._entities_registry, entity=entity)

            # 2.2) checking stop conditions
            stop_list = en.check_stop_conditions(
                self._entities_registry, en.StaticRegistry())
            self._stopped = self._stopped or len(stop_list) > 0
            for cond, entity in stop_list:
                info("[!!!] stop condition={}, entity={}"
                      "".format(cond.get_name(), str(entity)),
                      self.__class__.__name__)

            # 2.3) checking events num
            if max_events is not None \
                    and self._num_events_served >= max_events:
                info("[!T!] generated enough events: {}/{}".format(
                    self._num_events_served, max_events),
                    self.__class__.__name__)
                self._stopped = True

            if self._num_events_served % self.logging_interval == 0:
                if self.log_queue:
                    dqs = len(self._delayed_queue)
                    iqs = len(self._immediate_queue)
                    self.delayed_queue_size.append(dqs)
                    self.immediate_queue_size.append(iqs)
                    self.total_queue_size.append(dqs + iqs)
                if self.log_entities:
                    n_new = len(self._entities_registry.entities(
                        en.EntitiesRegistry.Stage.NEW))
                    n_inited = len(self._entities_registry.entities(
                        en.EntitiesRegistry.Stage.INITIALIZED))
                    n_finished = len(self._entities_registry.entities(
                        en.EntitiesRegistry.Stage.FINISHED))
                    self.entities_per_status_num[
                        en.EntitiesRegistry.Stage.NEW].append(n_new)
                    self.entities_per_status_num[
                        en.EntitiesRegistry.Stage.INITIALIZED].append(n_inited)
                    self.entities_per_status_num[
                        en.EntitiesRegistry.Stage.FINISHED].append(n_finished)
                    self.total_entities_num.append(n_new + n_inited +
                                                   n_finished)

            # 2.3) if still running, take the next event and process it
            if not self._stopped:
                if self._immediate_queue:
                    envelope = self._immediate_queue[0]
                    del self._immediate_queue[0]
                elif self._delayed_queue:
                    envelope = self._delayed_queue.pop()
                    self._time = envelope.time
                else:
                    envelope = None
                    self._stopped = True

                if envelope:
                    assert isinstance(envelope, Dispatcher.Envelope)
                    if envelope.target is not None:
                        # debug("handling event: source={}, target={},"
                        #       "event={}"
                        #       "".format(envelope.source, envelope.target,
                        #                 envelope.event),
                        #       sender=self.__class__.__name__)

                        assert self._entities_registry.contains(
                            envelope.target,
                            en.EntitiesRegistry.Stage.INITIALIZED)

                        handler = envelope.target.get_event_handler(
                            envelope.event, envelope.source)
                        handler(envelope.target, envelope.event,
                                envelope.source)
                    else:
                        # debug("handling event: source={}, event={}"
                        #       "".format(envelope.source, envelope.event),
                        #       sender=self.__class__.__name__)
                        handler = en.StaticRegistry().get_event_handler(
                            envelope.event, envelope.source)
                        handler(envelope.event, envelope.source)
                    self._num_events_served += 1

        # 3) completing the run by calling finalizers, if not aborted;
        #    otherwise, set all entities state as FINISHED without
        #    finalizing them.
        debug("finishing simulation", sender=self.__class__.__name__)
        self._state = Dispatcher.State.FINISHING
        if not self._aborted:
            en.finalize(self._entities_registry, en.StaticRegistry())
        else:
            entities = self._entities_registry.entities()
            for ent in entities:
                self._entities_registry.set_stage(
                    ent, entities.EntitiesRegistry.Stage.FINISHED)

        # ..) removing all entities detached during finalization
        self._state = Dispatcher.State.FINISHED
        for entity, op, abort in self._entities_cache:
            if op == 'remove':
                self._entities_registry.remove(entity)
        self._entities_cache.clear()
        fine("simulation finished", sender=self.__class__.__name__)

        if self.log_queue or self.log_entities:
            print("===== DISPATCHER: SIMULATION FINISHED")
        if self.log_queue:
            print("delayed queue size   : ", self.delayed_queue_size)
            print("immediate queue size : ", self.immediate_queue_size)
            print("total queue size     : ", self.total_queue_size)
        if self.log_entities:
            print("new entities num     : ", self.entities_per_status_num[
                en.EntitiesRegistry.Stage.NEW])
            print("inited entities num  : ", self.entities_per_status_num[
                en.EntitiesRegistry.Stage.INITIALIZED])
            print("finished entities num: ", self.entities_per_status_num[
                en.EntitiesRegistry.Stage.FINISHED])
            print("total entities num   : ", self.total_entities_num)

    def stop(self, abort=False):
        self._stopped = True
        self._aborted = abort

    def reset(self, keep_and_reset_entities=True):
        if en.CallStack():
            raise errors.ResetFromManagedFunction(en.CallStack())
        if self._state not in [Dispatcher.State.FINISHED,
                               Dispatcher.State.READY]:
            raise errors.DispatcherStateError(
                self._state.name, message="dispatcher reset in running state")

        self._state = Dispatcher.State.READY
        self._delayed_queue.clear()
        self._immediate_queue = []
        self._entities_cache = []
        self._stopped = False
        self._aborted = False
        self._time = 0.0
        self._next_event_index = itertools.count()
        self._num_events_served = 0

        if not keep_and_reset_entities:
            self._entities_registry.clear()
        else:
            entities = self._entities_registry.entities()
            for entity in entities:
                self._entities_registry.set_stage(
                    entity, en.EntitiesRegistry.Stage.NEW)


#######################################################################
# DISPATCHER ACCESSORS
#######################################################################

def run(max_events=None, queue=None):
    Dispatcher().reset(keep_and_reset_entities=True)
    Dispatcher().start(max_events=max_events, queue=queue)


def stop(abort=False):
    Dispatcher().stop(abort)


def reset():
    Dispatcher().reset(keep_and_reset_entities=False)


def add_entity(entity):
    Dispatcher().attach(entity)


def remove_entity(entity):
    Dispatcher().detach(entity)


def time():
    return Dispatcher().time


def set_model(model):
    Dispatcher().model = model


def get_model():
    return Dispatcher().model


def create_timeout(dt, event):
    dispatcher = Dispatcher()
    return dispatcher.schedule(
        event, dispatcher.time + dt, target=en.CallStack().entity)


def send_event(event, entity):
    dispatcher = Dispatcher()
    return dispatcher.schedule(event, target=entity)


def cancel(event_id):
    Dispatcher().cancel(event_id=event_id)


def get_num_events_served():
    return Dispatcher().num_events_served


#######################################################################
# ENVIRONMENT
#######################################################################

class LogLevel(Enum):
    ALL = -1
    TRACE = 0
    DEBUG = 1
    FINE = 2
    INFO = 3
    WARNING = 4
    ERROR = 5
    OFF = 6


class Environment(object, metaclass=Singleton):
    def __init__(self):
        self.log_level = LogLevel.INFO
        self.time_precision = 9
        self.sender_field_width = 16

    def log(self, level, message, sender=None):
        assert LogLevel.TRACE.value <= level.value <= LogLevel.ERROR.value
        if level.value >= self.log_level.value:
            s_time = "{time:0{width}.{precision}f} ".format(
                time=time(), width=self.time_precision+6,
                precision=self.time_precision)
            s_level = "[{level.name:^7s}] ".format(level=level)
            if sender is not None:
                s_sender = "({sender:^{width}s}) ".format(
                    sender=str(sender), width=self.sender_field_width)
            else:
                s_sender = ""
            print("{time}{level}{sender}{message}".format(
                time=s_time, level=s_level, sender=s_sender, message=message),
            flush=True)

    def trace_enter_function(self, fn, indention_level=0):
        self.log(LogLevel.TRACE, "{indent}----> {fun_name}".format(
            indent="  "*indention_level,
            fun_name=(fn.get_name() if hasattr(fn, 'get_name')
                      else fn.__name__)))

    def trace_exit_function(self, fn, indention_level=0):
        self.log(LogLevel.TRACE, "{indent}<---- {fun_name}".format(
            indent="  "*indention_level,
            fun_name=(fn.get_name() if hasattr(fn, 'get_name')
                      else fn.__name__)))

    def debug(self, message, sender=None):
        self.log(LogLevel.DEBUG, message, sender)

    def info(self, message, sender=None):
        self.log(LogLevel.INFO, message, sender)

    def fine(self, message, sender=None):
        self.log(LogLevel.FINE, message, sender)

    def warning(self, message, sender=None):
        self.log(LogLevel.WARNING, message, sender)

    def error(self, message, sender=None):
        self.log(LogLevel.ERROR, message, sender)
        stop(abort=True)


def debug(message, sender=None):
    Environment().debug(message, sender=sender)


def fine(message, sender=None):
    Environment().fine(message, sender=sender)


def info(message, sender=None):
    Environment().info(message, sender=sender)


def warning(message, sender=None):
    Environment().warning(message, sender=sender)


def error(message, sender=None):
    Environment().error(message, sender=sender)


def get_log_level():
    return Environment().log_level


def setup_env(log_level=LogLevel.DEBUG, sender_field_width=16,
              time_precision=9):
    env = Environment()
    env.log_level = log_level
    env.sender_field_width = sender_field_width
    env.time_precision = time_precision
