#include "System.h"
#include <sstream>


std::string Event::toString() const {
    std::stringstream ss;
    ss << "(Event: time=" << time
        << ", id=" << id
        << ", address=" << address
        << ", type=" << ::toString(type)
        << ")";
    return ss.str();
}


std::string toString(EventType value) {
    switch (value) {
        case STOP: return "STOP";
        case SOURCE_TIMEOUT: return "SOURCE_TIMEOUT";
        case SERVER_TIMEOUT: return "SERVER_TIMEOUT";
        default: return "???";
    }
}

System::System() {
    EventCmp cmp = [](Event *left, Event *right) {
        return (left->time > right->time) || (left->time == right->time && left->id > right->id);
    };
    eventsQueue = new std::priority_queue<Event*, std::vector<Event*>, EventCmp>(cmp);
}

System::~System() {
    while (!eventsQueue->empty()) {
        delete eventsQueue->top();
        eventsQueue->pop();
    }
    delete eventsQueue;
}


void System::schedule(EventType type, double interval, int address) {
    auto event = new Event{.id = nextId++, .time = this->time + interval, .address = address, .type = type};
    debug("\t- <SYSTEM> scheduled %s\n", event->toString().c_str());
    eventsQueue->push(event);
}


Event *System::nextEvent() {
    if (eventsQueue->empty()) {
        return new Event{.id = nextId++, .time = this->time, .address = -1, .type = STOP};
    }
    auto event = eventsQueue->top();
    eventsQueue->pop();
    debug("\t- <SYSTEM> extracted event %s\n", event->toString().c_str());
    this->time = event->time;
    return event;
}


void startService(Server *server, Packet *packet, System *system, Journal *journal) {
    auto time = system->getTime();
    auto address = server->getOwner()->getAddress();

    // Start service and schedule end:
    server->push(packet);
    double interval = server->getInterval();
    system->schedule(SERVER_TIMEOUT, interval, address);
    debug("\t- scheduled service end at %.3f (interval = %.3f)\n", time + interval, interval);

    // Update statistics:
    auto nodeRecords = journal->getRecords(address);
    packet->setServiceStartedAt(time);
    nodeRecords->waitTimes()->record(time - packet->getArrivedAt());
    nodeRecords->serverSize()->record(time, 1);
}


void handleArrival(Packet *packet, Node *node, System *system, Journal *journal) {
    auto server = node->getServer();
    auto queue = node->getQueue();
    auto time = system->getTime();
    auto address = node->getAddress();
    auto records = journal->getRecords(address);

    // Update number of arrived packets and arrival time:
    records->numPacketsArrived()->inc();
    packet->setArrivedAt(time);

    if (server->ready()) {
        // Server was empty: start service and record server size statistics:
        debug("\t- server was empty, start service\n");
        startService(server, packet, system, journal);
        records->serverSize()->record(time, server->size());
        records->systemSize()->record(time, node->size());
    } else if (queue->push(packet)) {
        // Packet was pushed: record queue and system size statistics:
        debug("\t- server was busy and queue wasn't full, pushing packet\n");
        records->queueSize()->record(time, queue->size());
        records->systemSize()->record(time, node->size());
    } else {
        // Packet was dropped: increment number of lost and dropped packets,
        // and delete the packet itself:
        debug("\t- server was busy and queue was full, dropping packet\n");
        records->numPacketsDropped()->inc();
        journal->getRecords(packet->getSource())->numPacketsLost()->inc();
        delete packet;
    }
}


void handleSourceTimeout(Node *node, System *system, Journal *journal) {
    debug("[%.3f] packet arrived at %d\n", system->getTime(), node->getAddress());
    auto source = node->getSource();
    auto address = node->getAddress();
    auto records = journal->getRecords(address);

    // Update statistics:
    records->numPacketsGenerated()->inc();
    journal->numPacketsGenerated()->inc();

    // Schedule next event, create the packet and start serving it:
    double interval = source->getInterval();
    system->schedule(SOURCE_TIMEOUT, interval, address);
    auto packet = source->createPacket(system->getTime());

    debug("\t- scheduled next arrival at %.3f (interval = %.3f)\n", system->getTime() + interval, interval);
    handleArrival(packet, node, system, journal);
    debug("\t- server size: %zd, queue size: %zd\n", node->getServer()->size(), node->getQueue()->size());
}


void handleServerTimeout(Node *node, System *system, Journal *journal) {
    auto address = node->getAddress();
    auto server = node->getServer();
    auto queue = node->getQueue();
    auto time = system->getTime();
    auto records = journal->getRecords(address);

    auto packet = server->pop();

    debug("[%.3f] server %d finished serving %s\n", time, address, packet->toString().c_str());

    // Update number of served packets and response time:
    records->numPacketsServed()->inc();
    records->responseTimes()->record(time - packet->getArrivedAt());
    records->departures()->record(time - server->getLastDepartureTime());
    server->setLastDepartureTime(time);

    // Decide, what to do with the packet:
    // - if its target is this node, deliver
    // - otherwise, forward to the next node
    if (packet->getTarget() == node->getAddress()) {
        // Packet was delivered: record statistics and delete packet
        debug("\t- packet was delivered\n");
        auto sourceRecords = journal->getRecords(packet->getSource());
        sourceRecords->numPacketsDelivered()->inc();
        sourceRecords->delays()->record(time - packet->getCreatedAt());
        delete packet;
    } else {
        // Packet should be forwarded to the next hop:
        debug("\t- forwarding packet to %d\n", node->getNextNode()->getAddress());
        handleArrival(packet, node->getNextNode(), system, journal);
    }

    // Check whether next packet can be served:
    if (!queue->empty()) {
        // Queue has packets: extract one, start serving it and record new queue size:
        debug("\t- getting next packet from the queue\n");
        packet = queue->pop();
        startService(server, packet, system, journal);
        records->queueSize()->record(time, queue->size());
    } else {
        // Queue is empty - record the server became ready:
        debug("\t- server is ready, queue is empty\n");
        records->serverSize()->record(time, 0);
    }

    // Update system size:
    debug("\t- server size: %zd, queue size: %zd\n", server->size(), queue->size());
    records->systemSize()->record(time, node->size());
}


void runMainLoop(Network *network, System *system, Journal *journal, int maxPackets) {
    // Initialize the model:
    debug("==== INIT ====\nnetwork: %s\n", network->toString().c_str());
    journal->reset(system->getTime());
    for (auto& addrNodePair: network->getNodes()) {
        auto node = addrNodePair.second;
        auto source = node->getSource();
        if (source) {
            auto address = addrNodePair.first;
            system->schedule(SOURCE_TIMEOUT, source->getInterval(), address);
        }
    }

    debug("==== RUN ====\n");
    while (!system->stopped()) {
        // Check whether enough packets were generated:
        if (maxPackets >= 0 && journal->numPacketsGenerated()->get() >= maxPackets) {
            system->stop();
            continue;
        }

        // If stop conditions not satisfied, extract the next event.
        auto event = system->nextEvent();
        if (event->type == STOP) {
            system->stop();
        } else if (event->type == SOURCE_TIMEOUT) {
            handleSourceTimeout(network->getNode(event->address), system, journal);
        } else if (event->type == SERVER_TIMEOUT) {
            handleServerTimeout(network->getNode(event->address), system, journal);
        }
        delete event;
    }
}
