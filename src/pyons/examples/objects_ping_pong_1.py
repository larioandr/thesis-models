import pyons
import itertools


class Client(pyons.Entity):
    def __init__(self, client_id, server=None, max_requests=3,
                 request_interval=1.0):
        super().__init__()
        self.ping_index = itertools.count()
        self.server = server
        self.client_id = client_id
        self.max_requests = max_requests
        self.num_requests_sent = 0
        self.num_responses_received = 0
        self.next_request_timeout_id = None
        self.request_interval = request_interval

    @pyons.Entity.initializer()
    def initialize(self):
        self.send_next_ping()

    @pyons.Entity.eventhandler(default=True)
    def handle_response(self, event, source):
        if not isinstance(source, Server):
            raise RuntimeError("expected events from server only")
        self.num_responses_received += 1
        pyons.info("received {} from the server".format(event), self)
        self.next_request_timeout_id = pyons.create_timeout(
            self.request_interval, "timeout")

    @pyons.Entity.eventhandler(guard=lambda ev, src: isinstance(src, Client))
    def handle_timeout(self, event, source):
        assert source is self and event == "timeout"
        self.send_next_ping()

    @pyons.Entity.death_condition()
    def check_requests_number(self):
        return self.num_requests_sent >= self.max_requests

    @pyons.Entity.finalizer()
    def finish(self):
        pyons.info("===== client id={} finished:"
                   "\n\t* requests sent     : {}/{}"
                   "\n\t* responses received: {}"
                   "".format(self.client_id, self.num_requests_sent,
                             self.max_requests, self.num_responses_received),
                   self)

    def send_next_ping(self):
        index = next(self.ping_index)
        pyons.info("sending ping #{} to the server".format(index), self)
        pyons.send_event(("ping", self.client_id, index), self.server)
        self.num_requests_sent += 1
        return index

    def __str__(self):
        return "Client-{}".format(self.client_id)


class Server(pyons.Entity):
    def __init__(self):
        super().__init__()
        self.num_requests_received = 0
        self.num_responses_sent = 0

    @pyons.Entity.eventhandler(guard=lambda ev, src: ev[0] == "ping")
    def handle_event(self, event, source):
        self.num_requests_received += 1
        self.num_responses_sent += 1
        msg = ("pong", event[1], event[2])
        pyons.info("received {}, sending {}".format(event, msg), "Server")
        pyons.send_event(msg, source)

    @pyons.Entity.finalizer(stage=1)
    def finish(self):
        pyons.info("===== server finished:"
                   "\n\t* requests received: {}"
                   "\n\t* responses sent   : {}"
                   "".format(self.num_requests_received,
                             self.num_responses_sent),
                   "Server")

    def __str__(self):
        return "Server"


class Generator(pyons.Entity):
    def __init__(self, generation_interval, server, max_requests,
                 request_interval, max_clients):
        super().__init__()
        self.timeout = generation_interval
        self.next_timeout_id = None
        self.next_client_id = 0
        self.request_interval = request_interval
        self.max_requests = max_requests
        self.server = server
        self.max_clients = max_clients

    @pyons.Entity.initializer(stage=0)
    def initialize(self):
        self.next_timeout_id = pyons.create_timeout(0, "timeout")

    @pyons.Entity.eventhandler(default=True)
    def generate(self, event, source):
        assert source is self and event == "timeout"
        client_id = self.next_client_id
        pyons.info("building a new client #{}".format(client_id), self)
        client = Client(client_id, self.server, self.max_requests,
                        self.request_interval)
        pyons.add_entity(client)
        self.next_timeout_id = pyons.create_timeout(self.timeout, "timeout")
        self.next_client_id += 1

    @pyons.Entity.stop_condition(lambda: pyons.get_num_events_served() > 10)
    def check_generated_enough_clients(self):
        return self.next_client_id >= self.max_clients

    @pyons.Entity.finalizer()
    def finish(self):
        pyons.info("===== generator finished:"
                   "\n\t* clients generated: {}".format(self.next_client_id))

    def __str__(self):
        return "Generator"


@pyons.stop_condition()
def check_simulation_time():
    return pyons.time() >= 100


if __name__ == '__main__':
    generation_interval = 2.0
    request_interval = 1.0
    max_requests = 5
    max_clients = 20

    server = Server()
    generator = Generator(generation_interval, server, max_requests,
                          request_interval, max_clients)

    pyons.add_entity(server)
    pyons.add_entity(generator)

    pyons.setup_env(pyons.LogLevel.DEBUG)

    pyons.run(1000)
