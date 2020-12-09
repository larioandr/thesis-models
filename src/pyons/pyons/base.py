class Singleton(type):
    """
    Metaclass for all singletons in the system (e.g. Kernel).
    The example code is taken from
    http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Metaprogramming.html
    """
    instance = None

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)

    def __call__(cls, *args, **kw):
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kw)
        return cls.instance
