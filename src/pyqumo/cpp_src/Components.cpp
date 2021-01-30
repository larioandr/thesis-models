#include "Components.h"

#include <sstream>
#include <utility>

// ==========================================================================
// Class Packet
// ==========================================================================
Packet::Packet(int source, int target, double createdAt) : source(source), target(target), createdAt(createdAt) {
}

Packet::~Packet() = default;

std::string Packet::toString() const {
    std::stringstream ss;
    ss << "(Packet: source=" << source << ", target=" << target << ", createdAt=" << createdAt << ")";
    return ss.str();
}


// ==========================================================================
// Class NodeComponent
// ==========================================================================
NodeComponent::NodeComponent() = default;
NodeComponent::~NodeComponent() = default;


// ==========================================================================
// Class Queue
// ==========================================================================
Queue::Queue(int capacity) : capacity(capacity) {}

Queue::~Queue() {
    while (!items.empty()) {
        delete items.front();
        items.pop();
    }
}

int Queue::push(Packet *packet) {
    if (full())
        return 0;
    items.push(packet);
    return 1;
}

Packet *Queue::pop() {
    if (items.empty())
        return nullptr;
    auto value = items.front();
    items.pop();
    return value;
}

std::string Queue::toString() const {
    std::stringstream ss;
    ss << "(Queue: size=" << items.size() << ", capacity="
        << (capacity < 0 ? "inf" : std::to_string(capacity))
        << ")";
    return ss.str();
}


// ==========================================================================
// Class Server
// ==========================================================================
Server::Server(DblFn intervalGetter) : intervalGetter(std::move(intervalGetter)) {}

Server::~Server() {
    delete packet;
}

int Server::push(Packet *pkt) {
    if (busy())
        return 0;
    this->packet = pkt;
    return 1;
}

Packet *Server::pop() {
    if (ready())
        return nullptr;
    auto value = this->packet;
    this->packet = nullptr;
    return value;
}

std::string Server::toString() const {
    std::stringstream ss;
    ss << "(Server: packet=" << (packet ? packet->toString() : "NULL") << ")";
    return ss.str();
}


// ==========================================================================
// Class Source
// ==========================================================================
Source::Source(const DblFn& intervalGetter, int destAddr, std::string  label)
: intervalGetter(intervalGetter), label(std::move(label)), destAddr(destAddr) {}

Source::~Source() = default;

Packet *Source::createPacket(double time) const {
    return new Packet(getOwner()->getAddress(), destAddr, time);
}

std::string Source::toString() const {
    std::stringstream ss;
    ss << "(Source: label=\"" << label << "\")";
    return ss.str();
}


// ==========================================================================
// Class Node
// ==========================================================================
Node::Node(int address, Queue *queue, Server *server, Source *source)
: address(address), queue(queue), server(server), source(source), nextNode(nullptr) {
    queue->setOwner(this);
    server->setOwner(this);
    if (source) {
        source->setOwner(this);
    }
}

Node::~Node() {
    delete queue;
    delete server;
    if (source) {
        delete source;
    }
}

std::string Node::toString() const {
    std::stringstream ss;
    ss << "(Node: address=" << address
        << ", server=" << server->toString()
        << ", queue=" << queue->toString()
        << ", nextNodeAddr=" << (nextNode ? std::to_string(nextNode->getAddress()) : "NULL")
        << ")";
    return ss.str();
}


// ==========================================================================
// Class Network
// ==========================================================================
Network::Network() = default;

Network::~Network() {
    for (auto& kv: nodes) {
        delete kv.second;
    }
}

std::string Network::toString() const {
    std::stringstream ss;
    ss << "(Network: nodes=[";
    for (auto& kv: nodes) {
        ss << "\n\t" << kv.first << ": " << kv.second->toString();
    }
    ss << "])";
    return ss.str();
}


// ==========================================================================
// Helpers
// ==========================================================================
Network *buildOneHopeNetwork(const DblFn& arrival, const DblFn& service, int queueCapacity) {
    auto queue = new Queue(queueCapacity);
    auto server = new Server(service);
    auto source = new Source(arrival, 0);
    auto node = new Node(0, queue, server, source);
    auto network = new Network;
    network->addNode(node);
    return network;
}
