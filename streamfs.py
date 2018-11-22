import numpy as np
import time
import psutil
import os
import math

'''
DESCRIPTION:
    Online Feature Selection Algorithm based on a paper by Wang et al. 2014. 
    This code is an adaptation of the official Matlab implementation
IN:
    x -> numpy array (datapoint)
    y -> numpy array (class of the datapoint: {-1;1})
    w -> numpy array (feature weights)
    num_features-> integer (number of features that should be returned)
OUT:
    w -> numpy array (updated feature weights)
    time -> float (computation time in seconds)
    memory -> float (currently used memory in percent of total physical memory)
'''
def _ofs(x, y, w, num_feature):
    start_t = time.process_time()

    eta = 0.2
    lamb = 0.01

    f = np.dot(w, x)  # prediction

    if y * f <= 1:  # update classifier w
        w = w + eta * y * x
        w = w * min(1, 1/(math.sqrt(lamb) * np.linalg.norm(w)))
        w = _truncate(w, num_feature)

    return w, time.process_time() - start_t, psutil.Process(os.getpid()).memory_percent()


'''
DESCRIPTION:
    Truncates a given array by setting all but the n biggest absolute values to zero.
IN:
    w -> numpy array (the array that should be truncated)
    num_features-> integer (number of features that should be kept)
OUT:
    w -> numpy array (truncated array)
'''
def _truncate(w, num_features):
    if len(w.nonzero()[0]) > num_features:
        w_sort_idx = np.argsort(abs(w))[-num_features:]
        zero_indices = [x for x in range(len(w)) if x not in w_sort_idx]
        w[zero_indices] = 0
    return w


'''
DESCRIPTION:
    Randomly sort the rows of a numpy array and extract the target variable Y and the features X
IN:
    data -> numpy array (dataset)
    target -> integer (index of the target variable)
    shuffle -> boolean (set to True if you want to sort the dataset randomly)
OUT:
    X -> numpy array (containing the features)
    Y -> numpy array (containing the target variable)
'''
def prepare_data(data, target, shuffle):
    if shuffle:
        np.random.shuffle(data)

    y = data[:, target]  # extract target variable
    x = np.delete(data, target, 1)  # delete target column

    return x, y


'''
DESCRIPTION:
    Iterate over all datapoints in a given matrix to simulate a data stream. 
    Perform given feature selection algorithm and return an array containing the weights for each (selected) feature
IN:
    X -> numpy array (dataset)
    Y -> numpy array (target)
    algorithm -> string (feature selection algorithm)
    num_features-> integer (number of features that should be returned)
OUT:
    ftr_weights -> numpy array (containing the weights of the (selected) features)
    stats -> python dictionary (contains i.a. average computation time in ms and 
             memory usage (in percent of physical memory) for one execution of the fs algorithm
'''
def simulate_stream(X, Y, algorithm, num_features):
    ftr_weights = np.zeros(X.shape[1], dtype=int)  # create empty feature weights array

    stats = {'memory_start': psutil.Process(os.getpid()).memory_percent(),  # get current memory usage of the process
             'time_measures': [],
             'memory_measures': [],
             'time_avg': 0,
             'memory_avg': 0}

    for i, x in enumerate(X):
        # OFS
        if algorithm == 'ofs':
            # update feature weights for every new data instance
            ftr_weights, time, memory = _ofs(x, Y[i], ftr_weights, num_features)

            # add difference in memory usage and computation time
            stats['memory_measures'].append(memory - stats['memory_start'])
            stats['time_measures'].append(time)
        else:
            print('Specified feature selection algorithm is not defined!')
            return ftr_weights, stats

    stats['time_avg'] = np.mean(stats['time_measures']) * 1000  # average time in milliseconds
    stats['memory_avg'] = np.mean(stats['memory_measures'])  # average percentage of used memory

    return ftr_weights, stats
