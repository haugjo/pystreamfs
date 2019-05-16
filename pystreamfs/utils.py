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

    # Test set = current batch
    x_test = X[i:i + param['batch_size'], selected_ftr]
    y_test = Y[i:i + param['batch_size']]

    # Training set = all samples except current batch
    if i == 0:
        # for first iteration st X_train = X_test
        x_train = x_test
        y_train = y_test
    else:
        x_train = X[0:i, selected_ftr]
        y_train = Y[0:i]

    # Train model
    model.fit(x_train, y_train)

    # Predict test set
    y_pred = model.predict(x_test)
    perf_score = metric(y_test, y_pred)

    return model, perf_score
