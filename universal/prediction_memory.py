import xxhash


class PredictionMemory(object):
    def __init__(self):
        self.memory = {}
        self.h = xxhash.xxh64()

    def set(self, k, v):
        self.h.update(k)
        self.memory[self.h.intdigest()] = v
        self.h.reset()

    def get(self, k):
        self.h.update(k)
        hash = self.h.intdigest()
        self.h.reset()
        return self.memory[hash] if hash in self.memory else None
