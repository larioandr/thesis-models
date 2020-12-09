import pyons
import itertools


class Model:
    def __init__(self):
        self.command_index = itertools.count()
        self.reply_timeout = 1.0
        self.next_ping_timeout = 4.0
        self.num_pings_generated = 0
        self.num_pongs_generated = 0


class Message(object):
    def __init__(self, name, index):
        self.name = name
        self.index = index

    def __str__(self):
        return "{}-{}".format(self.name, self.index)


@pyons.initializer(stage='1-base')
def initialize():
    pyons.info("PingPong is initializing", sender="initialize")
    cmd = Message('PING', next(pyons.get_model().command_index))
    pyons.get_model().num_pings_generated += 1
    pyons.create_timeout(0, cmd)


@pyons.eventhandler(lambda event, source: event.name[:4].lower() == 'ping')
def ping_handler(event, source=None):
    reply = Message('PONG', event.index)
    pyons.get_model().num_pongs_generated += 1
    pyons.create_timeout(pyons.get_model().reply_timeout, reply)
    pyons.info("received {}, sending {} in {} sec.".format(
        event, reply, pyons.get_model().reply_timeout), sender="ping_handler")


@pyons.eventhandler(lambda event, source: event.name[:4].lower() == 'pong')
def pong_handler(event, source=None):
    cmd = Message('PING', next(pyons.get_model().command_index))
    pyons.get_model().num_pings_generated += 1
    pyons.create_timeout(pyons.get_model().next_ping_timeout, cmd)
    pyons.info("received {}, sending {} in {} sec.".format(
        event, cmd, pyons.get_model().next_ping_timeout),
        sender="pong_handler")


@pyons.finalizer()
def finalize():
    m = pyons.get_model()
    pyons.info("PingPong is finished:", sender="finalize")
    print("\t+ pings generated:  {}".format(m.num_pings_generated))
    print("\t+ pongs generated:  {}".format(m.num_pongs_generated))
    print("\t+ events generated: {}".format(pyons.get_num_events_served()))

    events = ['{}@t={}'.format(envelope.event, envelope.time)
              for envelope in pyons.Dispatcher().events]
    print("\t+ resting events:   {}".format(events))


@pyons.stop_condition(guard=lambda: pyons.get_num_events_served() % 2 == 0)
def check_generated_enough():
    return pyons.get_model().num_pongs_generated >= 300


if __name__ == '__main__':
    model = Model()
    pyons.setup_env(log_level=pyons.LogLevel.DEBUG, sender_field_width=12,
                    time_precision=2)
    pyons.set_model(model)
    pyons.run(max_events=100)
