import numpy as np
import psutil
import os
import warnings
import time
from pystreamfs.plots import plot
from sklearn import preprocessing


def prepare_data(data, target, shuffle):
    """Extract the target and features

    :param numpy.nparray data: dataset
    :param int target: index of the target variable
    :param bool shuffle: set to True if you want to sort the dataset randomly
    :return: X (containing the features), Y (containing the target variable)
    :rtype: numpy.nparray, numpy.nparray
    """

    if shuffle:
        np.random.shuffle(data)

    Y = data[:, target]
    X = np.delete(data, target, 1)

    feature_names = np.zeros(X.shape[1])  # Todo: extract feature names

    return X, Y, feature_names


def create_data(n_samples, generator):
    """Generate the datasamples with the generator

    :param int : Generator you want to use,
    :return: generator
    :rtype: generator
    """

    X, y = generator.next_sample(n_samples)

    # Check if the stream has more data
    # generator.has_more_samples()

    return X, y


def simulate_stream(X, Y, generator, feature_selector, model, metric, param):
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
    # for i in range(0, X.shape[0], param['batch_size']):  # data stream Todo: switch between batch and generated data
    for i in range(0, param['max_timesteps'] * param['batch_size'], param['batch_size']):  # for generated data
        t = i / param['batch_size']  # time window
        param['t'] = t

        X, Y = create_data(param['batch_size'], generator.multiflow_alg)
        # Normalize
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)

        if 'feature_stream' in param and t in param['feature_stream']:  # feature stream
            ftr_indices = param['feature_stream'][t]
        elif 'feature_stream' not in param:
            ftr_indices = np.arange(0, X.shape[1])  # all features are available

        # Time taking
        start_tim = time.perf_counter()

        # Perform feature selection

        # Todo: switch between batch and generator
        # ftr_weights, feature_selector.prop = feature_selector.algorithm(X=X[i:i + param['batch_size'], ftr_indices], Y=Y[i:i + param['batch_size']], w=ftr_weights, fs_param=feature_selector.prop)
        ftr_weights, feature_selector.prop = feature_selector.algorithm(X=X[0:param['batch_size'], ftr_indices], Y=Y[0:param['batch_size']],
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


def plot_stats(stats, ftr_names, fs_algorithm, ml_model, metric, param, font_scale=1):  # Todo move to Pipeline or different file
    """Print statistics

    Prints performance metrics obtained during feature selection on simulated data stream

    :param dict stats: statistics
    :param np.ndarray ftr_names: names of original features
    :param dict param: parameters
    :param string fs_algorithm: name of the fs algorithm
    :param string ml_model: name of the ML model
    :param string metric: name of the performance metric
    :param float font_scale: factor by which the standard font size for text is scaled
    :return: chart
    :rtype: plt.figure
    """

    plot_data = dict()

    # Feature names & parameters
    plot_data['ftr_names'] = ftr_names
    plot_data['param'] = param
    plot_data['fs_algorithm'] = fs_algorithm
    plot_data['ml_model'] = ml_model
    plot_data['metric'] = metric
    plot_data['font_scale'] = font_scale

    # Time in ms
    plot_data['x_time'] = np.array(range(0, len(stats['time_measures'])))
    plot_data['y_time'] = np.array(stats['time_measures']) * 1000
    plot_data['avg_time'] = stats['time_avg'] * 1000

    # Memory in kB
    plot_data['x_mem'] = np.array(range(0, len(stats['memory_measures'])))
    plot_data['y_mem'] = np.array(stats['memory_measures']) / 1000
    plot_data['avg_mem'] = stats['memory_avg'] / 1000

    # Performance score
    plot_data['x_perf'] = np.array(range(0, len(stats['perf_measures'])))
    plot_data['y_perf'] = np.array(stats['perf_measures'])
    plot_data['avg_perf'] = stats['perf_avg']
    plot_data['q1_perf'] = np.percentile(stats['perf_measures'], 25, axis=0)
    plot_data['q3_perf'] = np.percentile(stats['perf_measures'], 75, axis=0)

    # Selected features
    plot_data['selected_ftr'] = stats['features']

    # Stability
    plot_data['x_stab'] = np.array(range(1, len(stats['stab_measures']) + 1))
    plot_data['y_stab'] = np.array(stats['stab_measures'])
    plot_data['avg_stab'] = stats['stab_avg']

    # Set ticks
    # X ticks
    plot_data['x_ticks'] = np.arange(0, plot_data['x_time'].shape[0], 1)
    if plot_data['x_time'].shape[0] > 30:  # plot every 5th x tick
        plot_data['x_ticks'] = ['' if i % 5 != 0 else b for i, b in enumerate(plot_data['x_ticks'])]

    # Y ticks for selected features
    plot_data['y_ticks_ftr'] = range(0, len(plot_data['ftr_names']))

    chart = plot(plot_data)

    return chart


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
