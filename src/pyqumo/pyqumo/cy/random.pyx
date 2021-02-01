from libcpp.vector cimport vector
import numpy as np


cdef class Rnd:
    cdef int index
    cdef vector[double] samples
    cdef int cacheSize
    cdef object fn

    def __cinit__(self, object fn, int cache_size = 10000):
        self.cacheSize = cache_size
        self.samples = vector[double](cache_size, 0.0)
        self.index = cache_size
        self.fn = fn

    def __init__(self, fn, cache_size=10000):
        pass

    # noinspection PyAttributeOutsideInit
    cdef double eval(self):
        cdef int cacheSize = self.cacheSize
        cdef int index = self.index
        cdef object fn = <object>self.fn
        if index >= cacheSize:
            self.samples = fn(cacheSize)
            self.index = 0
        x = self.samples[self.index]
        self.index += 1
        return x

    def __call__(self):
        return self.eval()

    def __repr__(self):
        return f"<CyRnd: ->"


cdef class Exp:
    cdef Rnd _rnd
    cdef double _rate

    def __cinit__(self, rate):
        self._rnd = Rnd(
            lambda size, r=rate: np.random.exponential(1/r, size=size))
        self._rate = rate

    cpdef double eval(self):
        return self._rnd.eval()

    def __call__(self, size = 1):
        return np.random.exponential(1/self._rate, size=size)
