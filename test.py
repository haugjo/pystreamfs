from memory_profiler import memory_usage
from time import sleep


def f(a=100):
    # a function that with growing
    # memory consumption
    b = a * 100
    sleep(.1)
    c = b * 100
    return a


if __name__ == '__main__':
    mem_usage = memory_usage((f, (10000000,)))
    print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
    print('Maximum memory usage: %s' % max(mem_usage))
