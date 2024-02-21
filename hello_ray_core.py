import ray
ray.init()


@ray.remote
def f(x):
    return x * x


@ray.remote
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n


futures1 = [f.remote(i) for i in range(4)]

counters = [Counter.remote() for i in range(4)]
[c.increment.remote() for c in counters]
futures2 = [c.read.remote() for c in counters]

print(ray.get(futures1))  # [0, 1, 4, 9]
print(ray.get(futures2))  # [1, 1, 1, 1]
