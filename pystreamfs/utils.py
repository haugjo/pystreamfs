import numpy as np


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
