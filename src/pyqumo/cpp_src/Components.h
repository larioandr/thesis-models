#ifndef COMPONENTS_H
#define COMPONENTS_H

#include "Base.h"
#include "Records.h"
#include "Functions.h"
#include <queue>
#include <functional>
#include <map>


class Packet : public Object {
public:
    Packet(int source, int target, double createdAt);
    ~Packet() override;

    inline int getSource() const { return source; }
    inline int getTarget() const { return target; }
    inline double getCreatedAt() const { return createdAt; }

    inline double getArrivedAt() const { return arrivedAt; }
    inline void setArrivedAt(double time) { arrivedAt = time; }

    inline double getServiceStartedAt() const { return serviceStartedAt; }
    inline void setServiceStartedAt(double time) { serviceStartedAt = time; }

    inline double getServiceFinishedAt() const { return serviceFinishedAt; }
    inline void setServiceFinishedAt(double time) { serviceFinishedAt = time; }

    std::string toString() const override;
private:
    int source;
    int target;
    double createdAt;
    double arrivedAt = 0.0;
    double serviceStartedAt = 0.0;
    double serviceFinishedAt = 0.0;
};


class Node;


class NodeComponent : public Object {
public:
    NodeComponent();
    ~NodeComponent() override;

    inline Node *getOwner() const { return owner; }
    inline void setOwner(Node *node) { this->owner = node; }
private:
    Node *owner = nullptr;
};


class Queue : public NodeComponent {
public:
    explicit Queue(int capacity = -1);
    ~Queue() override;

    int push(Packet* item);
    Packet* pop();

    inline ssize_t size() const { return items.size(); }
    inline bool empty() { return items.empty(); }
    inline int getCapacity() const { return capacity; }
    inline bool full() const { return capacity >= 0 && static_cast<int>(items.size()) >= capacity; }

    std::string toString() const override;
protected:
    std::queue<Packet*> items;
    int capacity;
};


class Server : public NodeComponent {
public:
    explicit Server(DblFn intervalGetter);
    ~Server() override;

    int push(Packet* pkt);
    Packet *pop();

    inline bool busy() const { return packet != nullptr; }
    inline bool ready() const { return packet == nullptr; }
    inline ssize_t size() const { return packet ? 1 : 0; }

    inline double getInterval() const { return intervalGetter(); }

    inline double getLastDepartureTime() const { return lastDepartureTime_; }
    inline void setLastDepartureTime(double time) { lastDepartureTime_ = time; }

    std::string toString() const override;
private:
    DblFn intervalGetter;
    Packet* packet = nullptr;
    double lastDepartureTime_ = 0.0;
};


class Source : public NodeComponent {
public:
    explicit Source(const DblFn& intervalGetter, int destAddr, std::string label = "");
    ~Source() override;

    inline double getInterval() const { return intervalGetter(); }
    inline int getDestAddr() const { return destAddr; }
    inline void setDestAddr(int addr) { destAddr = addr; }

    Packet *createPacket(double time) const;

    std::string toString() const override;
private:
    DblFn intervalGetter;
    std::string label;
    int destAddr;
};



class Node : public Object {
public:
    Node(int address, Queue *queue, Server *server, Source *source = nullptr);
    ~Node() override;

    inline int getAddress() const { return address; }

    inline Queue *getQueue() const { return queue; }
    inline Server *getServer() const { return server; }
    inline Source *getSource() const { return source; }

    inline void setNextNode(Node *node) { this->nextNode = node; }
    inline Node *getNextNode() const { return nextNode; }

    inline ssize_t size() const { return queue->size() + server->size(); }

    std::string toString() const override;

private:
    int address;
    Queue *queue;
    Server *server;
    Source *source;
    Node *nextNode;
};



class Network : public Object {
public:
    Network();
    ~Network() override;

    inline void addNode(Node *node) { nodes[node->getAddress()] = node; }
    inline Node *getNode(int address) const { return nodes.at(address); }

    inline const std::map<int, Node*>& getNodes() const { return nodes; }

    std::string toString() const override;
private:
    std::map<int, Node*> nodes;
};


Network *buildOneHopeNetwork(const DblFn& arrival, const DblFn& service, int queueCapacity);


#endif //COMPONENTS_H
