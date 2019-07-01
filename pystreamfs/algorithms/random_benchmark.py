import numpy as np


def run_random_benchmark(X, param, **kw):
    """Random Feature Selection

    Selects m random features indices. Can be used as benchmark or to justify feature selection.

    :param numpy.nparray X: current data batch
    :param numpy.nparray Y: labels of current batch
    :param numpy.nparray w: feature weights
    :param dict param: parameters, this includes...
        int num_features: number of selected features
    :return: w (feature weights), param
    :rtype numpy.ndarray, dict
    """

    ind = np.random.randint(0, X.shape[1], param['num_features'])
    w = np.zeros(X.shape[1])
    w[ind] = 1

    return w, param
