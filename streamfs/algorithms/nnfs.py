import time
import psutil
import os


def run_nnfs(x, y, param):
    """Neural Network Feature Selection

    Novel approach by Vadim Borisov

    :param numpy.nparray x: datapoint
    :param numpy.nparray y: class of the datapoint
    :param dict param: parameters

    :return: w (feature weights), time (computation time in seconds),
        memory (currently used memory in percent of total physical memory)
    :rtype numpy.ndarray, float, float

    .. warning: y must be -1 or 1

    .. todo @Vadim: feel free to change the name and description of your approach :)
    """

    start_t = time.perf_counter()  # time taking

    '''
    ADD YOUR CODE HERE
    '''

    w = []  # feature weights

    return w, time.perf_counter() - start_t, psutil.Process(os.getpid()).memory_percent()
