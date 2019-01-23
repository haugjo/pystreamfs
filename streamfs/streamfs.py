import numpy as np
import psutil
import os
import platform
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# import FS algorithms
from streamfs.algorithms.ofs import run_ofs
from streamfs.algorithms.fsds import run_fsds
from streamfs.algorithms.mcnn import run_mcnn, TimeWindow
from streamfs.algorithms.nnfs import run_nnfs
from streamfs.utils import comp_mfcr, perform_learning

# if on Unix system import resource module
if platform.system() == "Linux" or platform.system() == "Darwin":
    import resource


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

    # Do not display warnings in the console
    warnings.filterwarnings("ignore")

    ftr_weights = np.zeros(X.shape[1], dtype=int)  # create empty feature weights array
    model = None  # empty object that later holds the ML model
    mfcr = 0  # initialize mean feature change rate

    # measure current RAM usage in Byte
    if platform.system() == "Linux" or platform.system() == "Darwin":
        # on Unix
        start_memory = resource.getrusage(resource.RUSAGE_SELF)
    else:
        # on Windows
        # uss = “Unique Set Size”, this is the memory which is unique to a process and which would be freed if the process was terminated right now.
        start_memory = psutil.Process(os.getpid()).memory_full_info().uss

    stats = {'time_measures': [],
             'memory_measures': [],
             'acc_measures': [],
             'features': [],
             'mfcr_measures': [],
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
            ftr_weights, time = run_ofs(X[i:i+param['batch_size']], Y[i:i+param['batch_size']], ftr_weights, param['num_features'])

        # FSDS
        elif algorithm == 'fsds':
            x_t = X[i:i+param['batch_size']].T  # transpose x batch because FSDS assumes rows to represent features
            ftr_weights, time, param['b'], param['ell'] = run_fsds(param['b'], x_t, X.shape[1], param['k'], param['ell'])

        # MCNN
        elif algorithm == 'mcnn':
            ftr_weights, window, clusters, time = run_mcnn(X[i:i+param['batch_size']], Y[i:i+param['batch_size']], window, clusters, param)

        # NNFS
        elif algorithm == 'nnfs':
            ftr_weights, time = run_nnfs(X[i:i+param['batch_size']], Y[i:i+param['batch_size']], param)

        # no valid algorithm selected
        else:
            print('Specified feature selection algorithm is not defined!')
            return ftr_weights, stats

        # measure current memory consumption
        if platform.system() == "Linux" or platform.system() == "Darwin":  # on Unix
            memory = resource.getrusage(resource.RUSAGE_SELF)
        else:  # on Windows
            memory = psutil.Process(os.getpid()).memory_full_info().uss

        stats['memory_measures'].append(memory - start_memory)
        stats['time_measures'].append(time)

        # save indices of currently selected features
        selected_ftr = np.argsort(abs(ftr_weights))[::-1][:param['num_features']]
        stats['features'].append(selected_ftr.tolist())

        # perform actual learning
        model, acc = perform_learning(X, Y, i, selected_ftr, model, param)
        stats['acc_measures'].append(acc)

        # update mfcr for time windows t >= 1
        t = i/param['batch_size']
        if t >= 1:
            mfcr = comp_mfcr(stats['features'][-2], selected_ftr, X.shape[1], t, mfcr)
            stats['mfcr_measures'].append(mfcr)

    stats['time_avg'] = np.mean(stats['time_measures']) * 1000  # average time in milliseconds
    stats['memory_avg'] = np.mean(stats['memory_measures'])  # average memory usage
    stats['acc_avg'] = np.mean(stats['acc_measures']) * 100  # average accuracy score

    return ftr_weights, stats


def plot_stats(stats, ftr_names):
    """Print Time and Memory consumption

    Print the time, memory and accuracy measures as provided in stats. Also print the average time, memory consumption, accuracy

    :param dict stats: statistics
    :param np.array ftr_names: contains feature names (if included, features will be plotted
    :return: plt (plot containing 2 subplots for time and memory)
    """

    # plot time, memory and accuracy
    x_time = np.array(range(0, len(stats['time_measures'])))
    y_time = np.array(stats['time_measures'])*1000

    x_mem = np.array(range(0, len(stats['memory_measures'])))
    y_mem = np.array(stats['memory_measures']) / 1000

    x_acc = np.array(range(0, len(stats['acc_measures'])))
    y_acc = np.array(stats['acc_measures']) * 100
    acc_q1 = np.percentile(stats['acc_measures'], 25, axis=0) * 100
    acc_q3 = np.percentile(stats['acc_measures'], 75, axis=0) * 100

    x_mfcr = np.array(range(1, len(stats['mfcr_measures']) + 1))
    y_mfcr = np.array(stats['mfcr_measures'])

    plt.figure(figsize=(20, 25))
    gs1 = gridspec.GridSpec(5, 2)
    gs1.update(wspace=0.2, hspace=0.6)

    ax1 = plt.subplot(gs1[0, 0])
    ax1.plot(x_time, y_time)
    ax1.plot([0, x_time.shape[0]-1], [stats['time_avg'], stats['time_avg']])
    ax1.set_xlabel('t')
    ax1.set_ylabel('computation time (ms)')
    ax1.set_title('Time consumption for FS')
    ax1.legend(['time measures', 'avg. time'])

    ax2 = plt.subplot(gs1[0, 1])
    ax2.plot(x_mem, y_mem)
    ax2.plot([0, x_mem.shape[0]-1], [stats['memory_avg'] / 1000, stats['memory_avg'] / 1000])  # in kByte
    ax2.set_xlabel('t')
    ax2.set_ylabel('memory (kB)')
    ax2.set_title('Memory consumption for FS')
    ax2.legend(['memory measures', 'avg. memory'])

    ax3 = plt.subplot(gs1[1, :])
    ax3.plot(x_acc, y_acc)
    ax3.plot([0, x_acc.shape[0] - 1], [stats['acc_avg'], stats['acc_avg']])
    ax3.fill_between([0, x_mem.shape[0]-1], acc_q3, acc_q1, facecolor='green', alpha=0.5)
    ax3.set_xlabel('t')
    ax3.set_ylabel('accuracy (%)')
    ax3.set_title('Accuracy for FS')
    ax3.legend(['accuracy measures', 'mean',  'iqr'], loc="lower right")

    # plot selected features
    ftr_indices = range(0, len(ftr_names))

    ax4 = plt.subplot(gs1[2:-1, :])
    ax4.set_title('Selected features')
    ax4.set_ylabel('feature')
    ax4.set_xticklabels([])

    # plot selected features for each execution
    for i, val in enumerate(stats['features']):
        for v in val:
            ax4.scatter(i, v, marker='_', color='C0')

    if len(ftr_indices) <= 30:
        # if less than 30 features plot tic for each feature and change markers
        ax4.set_yticks(ftr_indices)
        ax4.set_yticklabels(ftr_names)

    # plot mfcr
    gs2 = gridspec.GridSpec(5, 2)
    gs2.update(hspace=0)

    ax5 = plt.subplot(gs2[4, :])
    ax5.plot(x_mfcr, y_mfcr)
    ax5.set_xlabel('t')
    ax5.set_ylabel('MFCR')
    ax5.legend(['MFCR development'], loc="lower right")

    return plt
