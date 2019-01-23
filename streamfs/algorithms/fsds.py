import numpy as np
import time
import numpy.linalg as ln


def run_fsds(B, Yt, m, k, ell=0):
    """Feature Selection on Data Streams

    Based on a paper by Huang et al. (2015). Feature Selection for unsupervised Learning.
    This code is copied from the Python implementation of the authors with minor reductions.

    :param numpy.ndarray B: sketched matrix (low-rank representation of all datapoints until current time)
    :param numpy.ndarray yt: m-by-n_t input matrix from data stream
    :param int m: number of original features
    :param int k: number of singular values (equal to number of clusters in the dataset)
    :param int ell: sketch size for a sketched m-by-ell matrix B


    :return: w (updated feature weights), time (computation time in seconds), B, ell
    :rtype numpy.ndarray, float, numpy.ndarray, int

    .. warning: fsds runs into a type error if n_t < 1000
    .. warning: features have to be equal to the rows in yt
    .. warning: yt has to contain only floats
    """

    start_t = time.perf_counter()  # time taking

    if ell < 1:
        ell = int(np.sqrt(m))

    if len(B) == 0:
        # for Y0, we need to first create an initial sketched matrix
        B = Yt[:, :ell]
        C = np.hstack((B, Yt[:, ell:]))
        n = Yt.shape[1] - ell
    else:
        # combine current sketched matrix with input at time t
        # C: m-by-(n+ell) matrix
        C = np.hstack((B, Yt))
        n = Yt.shape[1]

    U, s, V = ln.svd(C, full_matrices=False)
    U = U[:, :ell]
    s = s[:ell]
    V = V[:, :ell]

    # shrink step in Frequent Directions algorithm
    # (shrink singular values based on the squared smallest singular value)
    delta = s[-1] ** 2
    s = np.sqrt(s ** 2 - delta)

    # -- Extension of original code --
    # replace nan values with 0 to prevent division by zero error for small batch numbers
    s = np.nan_to_num(s)

    # update sketched matrix B
    # (focus on column singular vectors)
    B = np.dot(U, np.diag(s))

    # According to Section 5.1, for all experiments,
    # the authors set alpha = 2^3 * sigma_k based on the pre-experiment
    alpha = (2 ** 3) * s[k - 1]

    # solve the ridge regression by using the top-k singular values
    # X: m-by-k matrix (k <= ell)
    D = np.diag(s[:k] / (s[:k] ** 2 + alpha))

    # -- Extension of original code --
    # replace nan values with 0 to prevent division by zero error for small batch numbers
    D = np.nan_to_num(D)

    X = np.dot(U[:, :k], D)

    w = np.amax(abs(X), axis=1)

    return w, time.perf_counter() - start_t,  B, ell
