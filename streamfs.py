import numpy as np
import numpy.linalg as ln
import time
import psutil
import os
import math


def _ofs(x, y, w, num_features):
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
    """

    start_t = time.process_time()  # time taking

    eta = 0.2
    lamb = 0.01

    f = np.dot(w, x)  # prediction

    if y * f <= 1:  # update classifier w
        w = w + eta * y * x
        w = w * min(1, 1/(math.sqrt(lamb) * np.linalg.norm(w)))
        w = _truncate(w, num_features)

    return w, time.process_time() - start_t, psutil.Process(os.getpid()).memory_percent()


def _fsds(b, yt, m, k, ell=0):
    """Feature Selection on Data Streams

    Based on a paper by Huang et al. (2015). Feature Selection for unsupervised Learning.
    This code is copied from the Python implementation of the authors with minor reductions.

    :param numpy.ndarray b: sketched matrix (low-rank representation of all datapoints until current time)
    :param numpy.ndarray yt: m-by-n_t input matrix from data stream
    :param int m: number of original features
    :param int k: number of singular values (equal to number of clusters in the dataset)
    :param int ell: sketch size for a sketched m-by-ell matrix B


    :return: w (updated feature weights), time (computation time in seconds),
        memory (currently used memory in percent of total physical memory)
    :rtype numpy.ndarray, float, float

    .. warning: fsds runs into a type error if n_t < 1000
    .. warning: features have to be equal to the rows in yt
    .. warning: yt has to contain only floats
    .. todo: check why error occurs for different n_t
    """

    start_t = time.process_time()  # time taking

    if ell < 1:
        ell = int(np.sqrt(m))

    if len(b) == 0:
        # for Y0, we need to first create an initial sketched matrix
        B = yt[:, :ell]
        C = np.hstack((B, yt[:, ell:]))
        n = yt.shape[1] - ell
    else:
        # combine current sketched matrix with input at time t
        # C: m-by-(n+ell) matrix
        C = np.hstack((b, yt))
        n = yt.shape[1]

    U, s, V = ln.svd(C, full_matrices=False)
    U = U[:, :ell]
    s = s[:ell]
    V = V[:, :ell]

    # shrink step in Frequent Directions algorithm
    # (shrink singular values based on the squared smallest singular value)
    delta = s[-1] ** 2
    s = np.sqrt(s ** 2 - delta)

    # update sketched matrix B
    # (focus on column singular vectors)
    B = np.dot(U, np.diag(s))

    # According to Section 5.1, for all experiments,
    # the authors set alpha = 2^3 * sigma_k based on the pre-experiment
    alpha = (2 ** 3) * s[k - 1]

    # solve the ridge regression by using the top-k singular values
    # X: m-by-k matrix (k <= ell)
    D = np.diag(s[:k] / (s[:k] ** 2 + alpha))
    X = np.dot(U[:, :k], D)

    w = np.amax(abs(X), axis=1)

    return w, time.process_time() - start_t, psutil.Process(os.getpid()).memory_percent(),  b, ell


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


def prepare_data(data, target, shuffle):
    """Extract features X and target variable Y

    :param numpy.nparray data: dataset
    :param int target: index of the target variable
    :param bool shuffle: set to True if you want to sort the dataset randomly

    :return: X (containing the features), Y (containing the target variable)
    :rtype: numpy.nparray, numpy.nparray

    """

    if shuffle:
        np.random.shuffle(data)

    y = data[:, target]  # extract target variable
    x = np.delete(data, target, 1)  # delete target column

    return x, y


def simulate_stream(X, Y, algorithm, param):
    """Apply Feature Selection on stream data

    Iterate over all datapoints in a given matrix to simulate a data stream.
    Perform given feature selection algorithm and return an array containing the weights for each (selected) feature

    :param numpy.ndarray X: dataset
    :param numpy array Y: target
    :param str algorithm: feature selection algorithm
    :param int num_features: number of features that should be returned

    :return: ftr_weights (containing the weights of the (selected) features), stats (contains i.a. average computation
        time in ms and memory usage (in percent of physical memory) for one execution of the fs algorithm
    :rtype: numpy.ndarray, dict

    .. todo: enable OFS for different batch sizes
    """

    ftr_weights = np.zeros(X.shape[1], dtype=int)  # create empty feature weights array

    stats = {'memory_start': psutil.Process(os.getpid()).memory_percent(),  # get current memory usage of the process
             'time_measures': [],
             'memory_measures': [],
             'time_avg': 0,
             'memory_avg': 0}

    for i in range(0, X.shape[0], param['batch_size']):
        # OFS
        if algorithm == 'ofs':
            if param['batch_size'] == 1:
                ftr_weights, time, memory = _ofs(X[i], Y[i], ftr_weights, param['num_features'])
            else:
                print('WARNING: OFS currently only works for a batch size of 1!\n')
                return ftr_weights, stats
                # ftr_weights, time, memory = _ofs(X[i:i+param['batch_size']], Y[i:i+param['batch_size']], ftr_weights, param['num_features'])

        # FSDS
        elif algorithm == 'fsds':
            x_t = X[i:i+param['batch_size']].T  # transpose x batch because FSDS assumes rows to represent features
            ftr_weights, time, memory, param['b'], param['ell'] = _fsds(param['b'], x_t, X.shape[1], param['k'], param['ell'])

        # no valid algorithm selected
        else:
            print('Specified feature selection algorithm is not defined!')
            return ftr_weights, stats

        # add difference in memory usage and computation time
        stats['memory_measures'].append(memory - stats['memory_start'])
        stats['time_measures'].append(time)

    stats['time_avg'] = np.mean(stats['time_measures']) * 1000  # average time in milliseconds
    stats['memory_avg'] = np.mean(stats['memory_measures'])  # average percentage of used memory

    return ftr_weights, stats
