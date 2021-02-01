/**
 * @author Andrey Larionov
 */
#ifndef CQUMO_RANDOMS_H
#define CQUMO_RANDOMS_H

#include "Functions.h"
#include <random>

namespace cqumo {


void *createEngine();
void *createEngineWith(unsigned seed);
void destroyEngine(void *engine);


class RndBase {
  public:
    explicit RndBase(void *engine);
    virtual ~RndBase() = default;

    inline std::default_random_engine *engine() const { return engine_; }

    virtual double eval() = 0;
  private:
    std::default_random_engine *engine_ = nullptr;
};


class ExpVar : public RndBase {
  public:
    ExpVar(void *engine, double rate);
    ~ExpVar() override = default;

    double eval() override;
  private:
    std::exponential_distribution<double> distribution;
};

}

#endif //CQUMO_RANDOMS_H
