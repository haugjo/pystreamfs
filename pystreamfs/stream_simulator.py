import numpy as np
import psutil
import os
import warnings
import time
import seaborn as sns
from pystreamfs import visualizer as vis
from pystreamfs import live_visualizer as lv
from matplotlib import style
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pyplot import figure, draw, pause
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
from matplotlib.animation import TimedAnimation


def prepare_data(data, target, shuffle):
    """Extract the target and features

    :param numpy.nparray data: dataset
    :param int target: index of the target variable
    :param bool shuffle: set to True if you want to sort the dataset randomly
    :return: X (containing the features), Y (containing the target variable)
    :rtype: numpy.nparray, numpy.nparray
    """
    feature_names = list(data.drop(data.columns[target], axis=1).columns)
    data = np.array(data)

    if shuffle:
        np.random.shuffle(data)

    Y = data[:, target]
    X = np.delete(data, target, 1)

    return X, Y, feature_names


def simulate_stream(dataset, generator, feature_selector, model, metric, param):
    """Feature selection on simulated data stream

    Stream simulation by batch-wise iteration over dataset.
    Feature selection, classification and saving of performance metrics for every batch

    :param numpy.ndarray X: dataset
    :param numpy.ndarray Y: target
    :param function feature_selector: feature selection algorithm
    :param object model: Machine learning model for classification
    :param function metric: performance metric
    :param dict param: parameters
    :return: ftr_weights (selected features and their weights over time), stats (performance metrics over time)
    :rtype: numpy.ndarray, dict
    """

    # Do not display warnings in the console
    warnings.filterwarnings("ignore")

    ftr_weights = []  # empty feature weights array
    stats = {'time_measures': [],
             'memory_measures': [],
             'perf_measures': [],
             'features': [],
             'weights': [],
             'stab_measures': [],
             'time_avg': 0,
             'memory_avg': 0,
             'perf_avg': 0,
             'stab_avg': 0}

    # Stream simulation
    if dataset is not None:
        total_samples = dataset['X'].shape[0]
    else:  # if generator is defined
        total_samples = param['max_timesteps'] * param['batch_size']

    ###################################################################################
    # Set up window for plotting
    # Ex 1
    sns.set_context('paper')
    plt.style.use('seaborn-darkgrid')
    palette = ['#1f78b4', '#33a02c', '#fdbf6f', '#e31a1c']
    delay = 1

    # Lists for the plotted values
    # time
    x = []
    # time measures per timestep
    y_time = []
    # Mean time step
    y_time_mean = []
    # memory per timestep
    y_mem = []
    # Mean mem step
    y_mem_mean = []
    # best features
    y_features = []

    fig, axs = plt.subplots(3, 2, figsize=(10, 8))
    fig.canvas.set_window_title('Pystreamfs')
    plt.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.1)

    gs1 = gridspec.GridSpec(6, 2)
    gs1.update(wspace=0.2, hspace=0.6)

    # plt.rcParams.update({'font.size': 12 * param['font_scale']})
    # ax = fig.add_subplot(111)
    # line1, = ax.plot(x, y, 'r-')


    #########################################################################################
    # start data generation
    for i in range(0, total_samples, param['batch_size']):  # for generated data
        t = i / param['batch_size']  # time window
        param['t'] = t

        if dataset is not None:
            X = dataset['X']
            Y = dataset['Y']
            starting_idx = i
        else:  # if generator is defined
            X, Y = generator.create_sample(param['batch_size'])
            starting_idx = 0

        if 'feature_stream' in param and t in param['feature_stream']:  # feature stream
            ftr_indices = param['feature_stream'][t]
        elif 'feature_stream' not in param:
            ftr_indices = np.arange(0, X.shape[1])  # all features are available

        # Time taking
        start_tim = time.perf_counter()

        # Perform feature selection
        ftr_weights, feature_selector.prop = feature_selector.algorithm(X=X[starting_idx:starting_idx + param['batch_size'], ftr_indices], Y=Y[starting_idx:starting_idx + param['batch_size']],
                                          w=ftr_weights, fs_param=feature_selector.prop)

        selected_ftr = np.argsort(abs(ftr_weights))[::-1][:param['num_features']]  # top m absolute weights (features with highest influence)

        # Memory and time taking
        tim = time.perf_counter() - start_tim
        mem = psutil.Process(os.getpid()).memory_full_info().uss

        # Classify samples
        model, perf_score = classify(X, Y, i, selected_ftr, model, metric, param)

        # Save statistics
        stats['time_measures'].append(tim)
        stats['memory_measures'].append(mem)

        stats['features'].append(selected_ftr.tolist())
        stats['weights'].append(ftr_weights.tolist())
        stats['perf_measures'].append(perf_score)

        # stability measure for t >=1
        if t >= 1:
            stability = nogueira_stability(X.shape[1], stats['features'], param['r'])
            stats['stab_measures'].append(stability)

        # Life visualization
        if param['is_live']:


            # Append the data at each timestep to plot it afterwards
            x.append(t)
            y_time.append(stats['time_measures'][int(t)])
            y_time_mean.append(sum(y_time)/(t+1))
            y_mem.append(stats['memory_measures'][int(t)])
            y_mem_mean.append(sum(y_mem)/(t+1))

            # Call the different visualization plots (adapted from visualizer, methods in live_visualizer)
            lv.text_subplot(plt.subplot(gs1[0, :]), delay)
            # lv.regular_subplot(plt.subplot(gs1[1, 0]), x, y_time, 'Time $t$', 'Comp. Time (ms)', 'Time Consumption')
            lv.regular_subplot_mean(plt.subplot(gs1[1, 0]), x, y_time, 'Time $t$', 'Comp. Time (ms)', 'Time Consumption'
                                    , y_time_mean)
            # lv.regular_subplot(plt.subplot(gs1[1, 1]), x, y_mem, 'Time $t$', 'Memory usage (KB)', 'Memory usage')
            lv.regular_subplot_mean(plt.subplot(gs1[1, 1]), x, y_mem, 'Time $t$', 'Memory usage (KB)', 'Memory usage'
                               , y_mem_mean)

            plt.show(block=False)
            plt.pause(delay)
            # plt.close()

    # end of stream simulation

    # Compute average statistics
    stats['time_avg'] = np.mean(stats['time_measures'])  # average time in seconds
    stats['memory_avg'] = np.mean(stats['memory_measures'])  # average memory usage in Byte
    stats['perf_avg'] = np.mean(stats['perf_measures'])  # average performance metric
    stats['stab_avg'] = np.mean(stats['stab_measures'])  # average feature selection change rate

    # Add detections of concept drift
    # Todo: plot drifts + remove temporal parameters before ploting
    if 'check_drift' in param and param['check_drift'] is True:
        stats['drifts'] = param['drifts']
        stats['drift_ind_mean'] = param['drift_ind_mean']

        # Todo: needed only during experiments, can be removed afterwards
        stats['mu_measures'] = param['mu_measures']
        stats['sigma_measures'] = param['sigma_measures']

    return stats


def nogueira_stability(feature_space, selected_features, r):
    """
    Computation of stability score by Nogueira et al.
    (https://github.com/nogueirs/JMLR2018/tree/master/python)

    :param int feature space: number of original features
    :param list selected_features: selected features for all t <= current t
    :param int r: range of the shifting window
    :return: stability measure
    :rtype: float

    ..Todo.. Update README
    """
    # Construct Z
    Z = np.zeros([min(len(selected_features), r), feature_space])
    for row, col in enumerate(selected_features[-r:]):
        Z[row, col] = 1

    '''
    ORIGINAL CODE:
    Let us assume we have M>1 feature sets and d>0 features in total.
    This function computes the stability estimate as given in Definition 4 in  [1].

    INPUT: A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d).
           Each row of the binary matrix represents a feature set, where a 1 at the f^th position
           means the f^th feature has been selected and a 0 means it has not been selected.

    OUTPUT: The stability of the feature selection procedure
    '''

    M, d = Z.shape
    hatPF = np.mean(Z, axis=0)
    kbar = np.sum(hatPF)
    denom = (kbar / d) * (1 - kbar / d)
    return 1 - (M / (M - 1)) * np.mean(np.multiply(hatPF, 1 - hatPF)) / denom


def classify(X, Y, i, selected_ftr, model, metric, param):
    """Classify the samples of this batch

    :param numpy.ndarray X: dataset
    :param numpy.ndarray Y: target
    :param int i: current stream index (start of current batch)
    :param numpy.ndarray selected_ftr: indices of currently selected features
    :param object model: ML model
    :param object metric: performance metric
    :param dict param: parameters
    :return: model (ML model), perf_score(performance score)
    :rtype: object, float
    """

    # Test set = current batch OR last b samples (when dataset comes to an end)
    if i + param['batch_size'] > X.shape[0]:
        x_b = X[-param['batch_size']:, selected_ftr]
        y_b = Y[-param['batch_size']:]
    else:
        x_b = X[i:i + param['batch_size'], selected_ftr]
        y_b = Y[i:i + param['batch_size']]

    # Train if batch model
    if not hasattr(model, 'partial_fit'):  # model cannot be trained in online fashion
        if i == 0:
            x_train = x_b  # for first iteration st X_train = X_b
            y_train = y_b
        else:
            x_train = X[0:i, selected_ftr]  # Training set = all samples except current batch
            y_train = Y[0:i]

        model.fit(x_train, y_train)

        # Predict current batch (test set)
        y_pred = model.predict(x_b)

    # Train if online model
    elif hasattr(model, 'partial_fit'):  # model can be trained in online fashion
        if i == 0:  # for first iteration train with all features to get initial weights
            model.partial_fit(X[i:i + param['batch_size']], y_b, classes=np.unique(Y))  # must specify all classes at initialization, no posterior update of classes
            y_pred = model.predict(X[i:i + param['batch_size']])  # training error
        else:
            x_b_reshaped = np.zeros((x_b.shape[0], X.shape[1]))
            x_b_reshaped[:, selected_ftr] = x_b  # form train set where all but the selected features are zero
            y_pred = model.predict(x_b_reshaped)  # predict current batch

            # Train model
            model.partial_fit(x_b_reshaped, y_b)  # train model with current batch

    # Compute performance metric
    try:
        perf_score = metric(y_b, y_pred)
    except ValueError:
        perf_score = 0.5  # random performance
        print('Value error during computation of prediction metric!')

    return model, perf_score
