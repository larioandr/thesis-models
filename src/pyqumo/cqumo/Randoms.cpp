/**
 * @author Andrey Larionov
 */
#include "Randoms.h"
#include <chrono>
#include <algorithm>

namespace cqumo {

// Internal helpers
// ---------------------------------------------------------------------------
bool isNegative(double x) { return x <= 0; }
bool isNonPositive(double x) { return x < 0; }


// Functions
// ---------------------------------------------------------------------------
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


// Randoms
// --------------------------------------------------------------------------
Randoms::Randoms() 
: engine_(static_cast<std::default_random_engine*>(createEngine()))
{}

Randoms::Randoms(unsigned seed)
: engine_(static_cast<std::default_random_engine*>(createEngineWith(seed)))
{}

Randoms::~Randoms() {
    destroyEngine(static_cast<void*>(engine_));
}

RandomVariable *Randoms::createExponential(double rate) {
    return new ExponentialVariable(engine_, rate);
}

RandomVariable *Randoms::createUniform(double a, double b) {
    return new UniformVariable(engine_, a, b);
}

RandomVariable *Randoms::createNormal(double mean, double std) {
    return new NormalVariable(engine_, mean, std);
}

RandomVariable *Randoms::createErlang(int shape, double param) {
    return new ErlangVariable(engine_, shape, param);
}

RandomVariable *Randoms::createHyperExp(
        const std::vector<double>& rates, 
        const std::vector<double>& weights) {
    return new HyperExpVariable(engine_, rates, weights);
}


// RandomVariable
// ---------------------------------------------------------------------------
RandomVariable::RandomVariable(void *engine)
: engine_(static_cast<std::default_random_engine*>(engine)){}


// ExponentialVariable
// ---------------------------------------------------------------------------
ExponentialVariable::ExponentialVariable(void *engine, double rate)
: RandomVariable(engine), 
distribution(std::exponential_distribution<double>(rate))
{}

double ExponentialVariable::eval() {
    return distribution(*engine());
}

// UniformVarialbe
// ---------------------------------------------------------------------------
UniformVariable::UniformVariable(void *engine, double a, double b)
: RandomVariable(engine), 
distribution(std::uniform_real_distribution<double>(a, b))
{}

double UniformVariable::eval() {
    return distribution(*engine());
}


// NormalVariable
// ---------------------------------------------------------------------------
NormalVariable::NormalVariable(void *engine, double mean, double std)
: RandomVariable(engine),
distribution(std::normal_distribution<double>(mean, std))
{}

double NormalVariable::eval() {
    return distribution(*engine());
}


// HyperExpVariable
// --------------------------------------------------------------------------
HyperExpVariable::HyperExpVariable(
    void *engine, 
    const std::vector<double>& rates,
    const std::vector<double>& weights)
: RandomVariable(engine)
{
    // Validate arguments
    if (rates.size() != weights.size()) {
        throw std::runtime_error("rates and size weights mismatch");
    }
    if (std::any_of(weights.begin(),  weights.end(), isNegative)) {
        throw std::runtime_error("negative weight disallowed");
    }
    if (std::any_of(rates.begin(), rates.end(), isNonPositive)) {
        throw std::runtime_error("non-positive rate disallowed");
    }
    // Initialize fields
    choices_ = std::discrete_distribution<int>(weights.begin(), weights.end());
    for (auto& rate: rates) {
        exponents_.push_back(std::exponential_distribution<double>(rate));
    }
}

double HyperExpVariable::eval() {    
    auto state = static_cast<unsigned>(choices_(*engine()));
    return exponents_[state](*engine());
}


// ErlangVariable
// ---------------------------------------------------------------------------
ErlangVariable::ErlangVariable(void *engine, int shape, double param)
: RandomVariable(engine), 
shape_(shape), 
exponent(std::exponential_distribution<double>(param)) 
{}

double ErlangVariable::eval() {
    auto enginePtr = engine();
    double value = 0.0;
    for (int i = 0; i < shape_; i++) {
        value += exponent(*enginePtr);
    }
    return value;
}


}
