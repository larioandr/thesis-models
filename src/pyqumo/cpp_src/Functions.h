#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <functional>

typedef std::function<double()> DblFn;
typedef std::function<double(void*)> CtxDblFn;


class ContextFunctor {
public:
    explicit ContextFunctor(const CtxDblFn& fn, void *context) : fn_(fn), context_(context) {} // NOLINT(modernize-pass-by-value)
    ContextFunctor(const ContextFunctor& other) = default;
    ContextFunctor& operator=(const ContextFunctor& other) = default;

    double operator()() const {
        return fn_(context_);
    }
private:
    CtxDblFn fn_;
    void *context_;
};


#endif //FUNCTIONS_H
