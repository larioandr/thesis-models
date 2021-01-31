from libcpp.vector cimport vector


cdef class CyRnd:
    cdef int _index
    cdef vector[double] _samples
    cdef int _cacheSize
    cdef object _fn

    def __init__(self, fn, cache_size=10000):
        # Initialize C++ fields:
        self._fn = fn
        self._cacheSize = cache_size
        self._samples = vector[double](cache_size, 0.0)
        self._index = cache_size

    def __call__(self):
        cdef int cacheSize = self._cacheSize
        cdef int index = self._index
        if index >= cacheSize:
            self._samples = self._fn(cacheSize)
            self._index = 0
        x = self._samples[self._index]
        self._index += 1
        return x

    def __repr__(self):
        return f"<CyRnd: ->"
