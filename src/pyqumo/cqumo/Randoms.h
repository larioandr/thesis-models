/**
 * @author Andrey Larionov
 */
#ifndef CQUMO_RANDOMS_H
#define CQUMO_RANDOMS_H

#include "Functions.h"
#include <random>
#include <vector>

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


class UniformVar : public RndBase {
  public:
    UniformVar(void *engine, double a, double b);
    ~UniformVar() override = default;

    double eval() override;
  private:
    std::uniform_real_distribution<double> distribution;
};


class NormalVar : public RndBase  {
  public:
    NormalVar(void *engine, double mean, double std);
    ~NormalVar() override = default;

    double eval() override;
  private:
    std::normal_distribution<double> distribution;
};


class ErlangVar : public RndBase {
  public:
    ErlangVar(void *engine, int shape, double param);
    ~ErlangVar() override = default;

    double eval() override;
  private:
    std::exponential_distribution<double> exponent;
};


class HyperExpVar : public RndBase {
  public:
    HyperExpVar(
      void *engine, 
      const std::vector<double>& rates,
      const std::vector<double>& probs);

    ~HyperExpVar() override = default;

    double eval() override;
  private:
    std::vector<std::exponential_distribution<double>> exponents_;
    std::vector<double> probs_;
};



}

#endif //CQUMO_RANDOMS_H
