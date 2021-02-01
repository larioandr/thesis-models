/**
 * @author Andrey Larionov
 */
#include "Randoms.h"
#include <chrono>

namespace cqumo {

void *createEngine() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    return createEngineWith(seed);
}

void *createEngineWith(unsigned seed) {
    return static_cast<void*>(new std::default_random_engine(seed));
}

void destroyEngine(void *engine) {
    delete static_cast<std::default_random_engine*>(engine);
}


RndBase::RndBase(void *engine)
: engine_(static_cast<std::default_random_engine*>(engine)){}

ExpVar::ExpVar(void *engine, double rate)
: RndBase(engine), distribution(std::exponential_distribution<double>(rate))
{}

double ExpVar::eval() {
    return distribution(*engine());
}

}
