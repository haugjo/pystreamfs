import numpy as np
import math


def run_ofs(X, Y, w, param):
    """Online Feature Selection

    Based on a paper by Wang et al. 2014. Feature Selection for binary classification.
    This code is an adaptation of the official Matlab implementation.

    :param numpy.nparray X: current data batch
    :param numpy.nparray Y: labels of current batch
    :param numpy.nparray w: feature weights
    :param dict param: parameters, this includes...
        int num_features: number of selected features
    :return: w (feature weights), param
    :rtype numpy.ndarray, dict
    """

    eta = 0.2
    lamb = 0.01

    for x, y in zip(X, Y):  # perform feature selection for each instance in batch
        # Convert label to -1 and 1
        y = -1 if y == 0 else 1

        f = np.dot(w, x)  # prediction

        if y * f <= 1:  # update classifier w
            w = w + eta * y * x
            w = w * min(1, 1/(math.sqrt(lamb) * np.linalg.norm(w)))
            w = _truncate(w, param['num_features'])

    return w, param


def _truncate(w, num_features):
    """Truncate the weight vector

    Set all but the **num_features** biggest absolute values to zero.

    :param numpy.nparray w: weights
    :param int num_features: number of features that should be kept
    :return: w (truncated array)
    :rtype: numpy.nparray
    """

    if len(w.nonzero()[0]) > num_features:
        w_sort_idx = np.argsort(abs(w))[-num_features:]
        zero_indices = [x for x in range(len(w)) if x not in w_sort_idx]
        w[zero_indices] = 0
    return w
