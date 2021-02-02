import numpy as np
from pyqumo.cqumo.randoms cimport createEngine, createEngineWith


cdef class Engine:
    cdef void* cEngine

    def __init__(self, seed=None):
        if seed is None:
            self.cEngine = createEngine()
        else:
            self.cEngine = createEngineWith(<unsigned>seed)

    def __dealloc__(self):
        destroyEngine(self.cEngine)

    cdef void* getEngine(self):
        return self.cEngine


cdef class ExpGen:
    cdef ExpVar *cExpVar
    cdef void* cEngine

    def __cinit__(self, double rate):
        cEngine = createEngine()
        self.cExpVar = new ExpVar(cEngine, rate)

    def __init__(self, rate):
        pass
        
    def __dealloc__(self):
        del self.cExpVar
        destroyEngine(self.cEngine)

    cpdef double eval(self):
        return self.cExpVar.eval()

    def __call__(self, size=1):
        if size == 1:
            return self.eval()
        return np.asarray([self.eval() for _ in range(size)])


def createExpGen(rate):
    return ExpGen(rate)
