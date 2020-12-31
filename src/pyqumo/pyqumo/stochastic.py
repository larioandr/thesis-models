class Rnd:
    def __init__(self, fn, cache_size=10000, label=""):
        self.__fn = fn
        self.__cache_size = cache_size
        self.__samples = []
        self.__index = cache_size
        self.label = label

    def __call__(self):
        if self.__index >= self.__cache_size:
            self.__samples = self.__fn(self.__cache_size)
            self.__index = 0
        x = self.__samples[self.__index]
        self.__index += 1
        return x

    def __repr__(self):
        return f"<Rnd: '{self.label}'>"
