import numpy as np
import math


def run_ofs(X, Y, w, param):
    """Online Feature Selection Algorithm

    Based on a paper by Wang et al. 2014. Feature Selection for binary classification.
    This code is an adaptation of the official Matlab implementation.

    :param numpy.nparray X: data for current batch
    :param numpy.nparray Y: classes of the datapoints in current batch
    :param numpy.nparray w: feature weights
    :param int num_features: number of features that should be returned

    :return: w (updated feature weights), time (computation time in seconds)
    :rtype numpy.ndarray, float

    .. warning: y must be -1 or 1
    """

    eta = 0.2
    lamb = 0.01

    for x, y in zip(X, Y):  # perform feature selection for each instance in batch
        f = np.dot(w, x)  # prediction

        if y * f <= 1:  # update classifier w
            w = w + eta * y * x
            w = w * min(1, 1/(math.sqrt(lamb) * np.linalg.norm(w)))
            w = _truncate(w, param['num_features'])

    return w, param


def _truncate(w, num_features):
    """Truncates a given array

    Set all but the **num_features** biggest absolute values to zero.

    :param numpy.nparray w: the array that should be truncated
    :param int num_features: number of features that should be kept

    :return: w (truncated array)
    :rtype: numpy.nparray

    """

    if len(w.nonzero()[0]) > num_features:
        w_sort_idx = np.argsort(abs(w))[-num_features:]
        zero_indices = [x for x in range(len(w)) if x not in w_sort_idx]
        w[zero_indices] = 0
    return w
