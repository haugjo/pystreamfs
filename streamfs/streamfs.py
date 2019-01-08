import numpy as np
import psutil
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# import FS algorithms
from streamfs.algorithms.ofs import run_ofs
from streamfs.algorithms.fsds import run_fsds
from streamfs.algorithms.mcnn import run_mcnn, TimeWindow
from streamfs.algorithms.nnfs import run_nnfs


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
    :param dict param: parameters for feature selection

    :return: ftr_weights (containing the weights of the (selected) features), stats (contains i.a. average computation
        time in ms and memory usage (in percent of physical memory) for one execution of the fs algorithm
    :rtype: numpy.ndarray, dict
    """

    ftr_weights = np.zeros(X.shape[1], dtype=int)  # create empty feature weights array
    model = None  # empty object that later holds the ML model

    stats = {'memory_start': psutil.Process(os.getpid()).memory_percent(),  # get current memory usage of the process
             'time_measures': [],
             'memory_measures': [],
             'acc_measures': [],
             'features': [],
             'time_avg': 0,
             'memory_avg': 0,
             'acc_avg': 0}

    # For MCNN only
    if algorithm == 'mcnn':
        window = TimeWindow(X[0])  # create a time window object
        clusters = dict()  # create an empty dict of clusters

    for i in range(0, X.shape[0], param['batch_size']):
        # Add additional elif statement for new algorithms

        # OFS
        if algorithm == 'ofs':
            ftr_weights, time, memory = run_ofs(X[i:i+param['batch_size']], Y[i:i+param['batch_size']], ftr_weights, param['num_features'])

        # FSDS
        elif algorithm == 'fsds':
            x_t = X[i:i+param['batch_size']].T  # transpose x batch because FSDS assumes rows to represent features
            ftr_weights, time, memory, param['b'], param['ell'] = run_fsds(param['b'], x_t, X.shape[1], param['k'], param['ell'])

        # MCNN
        elif algorithm == 'mcnn':
            ftr_weights, window, clusters, time, memory = run_mcnn(X[i:i+param['batch_size']], Y[i:i+param['batch_size']], window, clusters, param)

        # NNFS
        elif algorithm == 'nnfs':
            ftr_weights, time, memory = run_nnfs(X[i:i+param['batch_size']], Y[i:i+param['batch_size']], param)

        # no valid algorithm selected
        else:
            print('Specified feature selection algorithm is not defined!')
            return ftr_weights, stats

        # add difference in memory usage and computation time
        stats['memory_measures'].append(memory - stats['memory_start'])
        stats['time_measures'].append(time)

        # save indices of currently selected features
        selected_ftr = np.argsort(abs(ftr_weights))[::-1][:param['num_features']]
        stats['features'].append(selected_ftr)

        # perform actual learning
        model, acc = perform_learning(X, Y, i, selected_ftr, model, param)
        stats['acc_measures'].append(acc)

    stats['time_avg'] = np.mean(stats['time_measures']) * 1000  # average time in milliseconds
    stats['memory_avg'] = np.mean(stats['memory_measures']) * 100  # average percentage of used memory
    stats['acc_avg'] = np.mean(stats['acc_measures']) * 100  # average accuracy score

    return ftr_weights, stats


def perform_learning(X, y, i, selected_ftr, model, param):
    # test samples = current batch
    X_test = X[i:i + param['batch_size'], selected_ftr]
    y_test = y[i:i + param['batch_size']]

    # training samples = all samples up until current batch
    if i == 0:
        # for first iteration st X_train = X_test
        X_train = X_test
        y_train = y_test
    else:
        X_train = X[0:i, selected_ftr]
        y_train = y[0:i]

    if model is None and param['algorithm'] == "knn":
        model = KNeighborsClassifier()
    elif model is None and param['algorithm'] == "tree":
        model = DecisionTreeClassifier(random_state=0)

    # set n_neighbors for KNN
    if type(model) is KNeighborsClassifier and X_train.shape[0] < param['neighbors']:
        # adjust KNN neighbors if there are too less samples available yet
        model.n_neighbors = X_train.shape[0]
    else:
        model.n_neighbors = param['neighbors']

    # train model
    model.fit(X_train, y_train)

    # predict current batch
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc


def plot_stats(stats, ftr_names):
    """Print Time and Memory consumption

    Print the time and memory measures as provided in stats. Also print the average time and memory consumption

    :param dict stats: statistics
    :param np.array ftr_names: contains feature names (if included, features will be plotted
    :return: plt (plot containing 2 subplots for time and memory)
    """

    # plot time and memory
    x_time = np.array(range(0, len(stats['time_measures'])))
    y_time = np.array(stats['time_measures'])*1000

    x_mem = np.array(range(0, len(stats['memory_measures'])))
    y_mem = np.array(stats['memory_measures'])*100

    plt.figure(figsize=(15, 25))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.subplot2grid((3, 2), (0, 0))
    plt.plot(x_time, y_time)
    plt.plot([0, x_time.shape[0]-1], [stats['time_avg'], stats['time_avg']])
    plt.xlabel('t')
    plt.ylabel('computation time (ms)')
    plt.title('Time consumption for FS')
    plt.legend(['time measures', 'avg. time'])

    plt.subplot2grid((3, 2), (0, 1))
    plt.plot(x_mem, y_mem)
    plt.plot([0, x_mem.shape[0]-1], [stats['memory_avg'], stats['memory_avg']])
    plt.xlabel('t')
    plt.ylabel('memory (% of RAM)')
    plt.title('Memory consumption for FS')
    plt.legend(['memory measures', 'avg. memory'])

    # plot selected features
    ftr_indices = range(0, len(ftr_names))

    plt.subplot2grid((3, 2), (1, 0), rowspan=2, colspan=2)
    plt.title('Selected features')
    plt.xlabel('t')
    plt.ylabel('feature')

    # plot selected features for each execution
    for i, val in enumerate(stats['features']):
        for v in val:
            plt.scatter(i, v, marker='_', color='C0')

    # markers indicating final list of features
    marker_y = '_'
    marker_n = '_'

    if len(ftr_indices) <= 30:
        # if less than 30 features plot tic for each feature and change markers
        plt.yticks(ftr_indices, ftr_names)
        marker_y = 'P'
        marker_n = 'x'

    # plot final set of features
    for i in ftr_indices:
        if i in stats['features'][-1]:
            plt.scatter(len(stats['features']), i, marker=marker_y, color="C2")
        else:
            plt.scatter(len(stats['features']), i, marker=marker_n, color="C3")

    return plt
