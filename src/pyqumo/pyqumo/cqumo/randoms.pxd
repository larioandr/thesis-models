cdef extern from "Randoms.h" namespace "cqumo":
    cdef void* createEngine()
    cdef void* createEngineWith(unsigned seed)
    cdef void destroyEngine(void* engine)

    cdef cppclass ExpVar:
        ExpVar(void *engine, double rate)
        double eval()
