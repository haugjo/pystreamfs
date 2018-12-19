import numpy as np
import time
import psutil
import os
import math
from streamfs.utils import truncate


def run_ofs(x, y, w, num_features):
    """Online Feature Selection Algorithm

    Based on a paper by Wang et al. 2014. Feature Selection for binary classification.
    This code is an adaptation of the official Matlab implementation.

    :param numpy.nparray x: datapoint
    :param numpy.nparray y: class of the datapoint
    :param numpy.nparray w: feature weights
    :param int num_features: number of features that should be returned

    :return: w (updated feature weights), time (computation time in seconds),
        memory (currently used memory in percent of total physical memory)
    :rtype numpy.ndarray, float, float

    .. warning: y must be -1 or 1

    .. todo: enable OFS for different batch sizes
    """

    start_t = time.perf_counter()  # time taking

    eta = 0.2
    lamb = 0.01

    f = np.dot(w, x)  # prediction

    if y * f <= 1:  # update classifier w
        w = w + eta * y * x
        w = w * min(1, 1/(math.sqrt(lamb) * np.linalg.norm(w)))
        w = truncate(w, num_features)

    return w, time.perf_counter() - start_t, psutil.Process(os.getpid()).memory_percent()
