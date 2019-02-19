import numpy as np
import numpy.linalg as ln


def run_fsds(X, param, **kw):
    """Feature Selection on Data Streams

    Based on a paper by Huang et al. (2015). Feature Selection for unsupervised Learning.
    This code is copied from the Python implementation of the authors with minor reductions and adaptations.

    :param numpy.ndarray X: current data batch
    :param dict param: parameters, this includes...
        numpy.ndarray B: sketched matrix (low-rank representation of all datapoints until current time)
        int m: number of original features
        int k: number of singular values (equal to number of clusters in the dataset)
        int ell: sketch size for a sketched m-by-ell matrix B
    :return: w (feature weights), param (with updated B and ell)
    :rtype numpy.ndarray, dict

    .. warning: features are represented as rows in Yt
    .. warning: Yt has to contain only floats
    """

    Yt = X.T  # algorithm assumes rows to represent features

    if param['ell'] < 1:
        param['ell'] = int(np.sqrt(param['m']))

    if len(param['B']) == 0:
        # for Y0, we need to first create an initial sketched matrix
        param['B'] = Yt[:, :param['ell']]
        C = np.hstack((param['B'], Yt[:, param['ell']:]))
        n = Yt.shape[1] - param['ell']
    else:
        # combine current sketched matrix with input at time t
        # C: m-by-(n+ell) matrix
        C = np.hstack((param['B'], Yt))
        n = Yt.shape[1]

    U, s, V = ln.svd(C, full_matrices=False)
    U = U[:, :param['ell']]
    s = s[:param['ell']]
    V = V[:, :param['ell']]

    # shrink step in Frequent Directions algorithm
    # (shrink singular values based on the squared smallest singular value)
    delta = s[-1] ** 2
    s = np.sqrt(s ** 2 - delta)

    # -- Extension of original code --
    # replace nan values with 0 to prevent division by zero error for small batch numbers
    s = np.nan_to_num(s)

    # update sketched matrix B
    # (focus on column singular vectors)
    param['B'] = np.dot(U, np.diag(s))

    # According to Section 5.1, for all experiments,
    # the authors set alpha = 2^3 * sigma_k based on the pre-experiment
    alpha = (2 ** 3) * s[param['k'] - 1]

    # solve the ridge regression by using the top-k singular values
    # X: m-by-k matrix (k <= ell)
    D = np.diag(s[:param['k']] / (s[:param['k']] ** 2 + alpha))

    # -- Extension of original code --
    # replace nan values with 0 to prevent division by zero error for small batch numbers
    D = np.nan_to_num(D)

    X = np.dot(U[:, :param['k']], D)

    w = np.amax(abs(X), axis=1)

    return w,  param
