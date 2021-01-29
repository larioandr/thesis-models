#ifndef SYSTEM_H_
#define SYSTEM_H_

#include "Records.h"
#include "Components.h"
#include <queue>


enum EventType {
    STOP = 0,
    SOURCE_TIMEOUT = 1,
    SERVER_TIMEOUT = 2
};

std::string toString(EventType eventType);


struct Event {
    unsigned id;
    double time;
    int address;
    EventType type;

    std::string toString() const;
};


typedef std::function<bool(Event*, Event*)> EventCmp;


class System {
public:
    explicit System();
    virtual ~System();

    void schedule(EventType event, double interval, int address);
    Event *nextEvent();

    inline double getTime() const { return time; }

    inline bool stopped() const { return wasStopped; }
    inline void stop() { wasStopped = true; }
private:
    std::priority_queue<Event*, std::vector<Event*>, EventCmp> *eventsQueue = nullptr;
    double time = 0.0;
    unsigned nextId = 0;
    bool wasStopped = false;
};


void startService(Server *server, Packet *packet, System *system, Journal *journal);
void handleArrival(Packet *packet, Node *node, System *system, Journal *journal);
void handleSourceTimeout(Node *node, System *system, Journal *journal);
void handleServerTimeout(Node *node, System *system, Journal *journal);
void runMainLoop(Network *network, System *system, Journal *journal, int maxPackets = -1);


#endif //SYSTEM_H_
