import numpy as np


def truncate(w, num_features):
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


def comp_mfcr(ftr_t_1, ftr_t, total_no_ftr, t, mfcr):
    # number of changes with regard to t-1
    # every change in the selected feature set indicates 2 changes of feature weights -> thus we multiply by 2
    c = len(set(ftr_t_1).difference(set(ftr_t))) * 2

    if t == 1:
        return c / len(ftr_t) * t
    else:
        return mfcr * (t-1)/t + c / (total_no_ftr * t)
