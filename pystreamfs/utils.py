from sklearn.metrics import accuracy_score


def fscr_score(ftr_t_1, ftr_t, n):
    """Feature Selection Change Rate

    The percentage of selected features that changed with respect to the previous time window

    :param ftr_t_1: selected features in t-1
    :param ftr_t: selected features in t (current time window)
    :param n: number of selected features
    :return: fscr
    :rtype: float
    """
    c = len(set(ftr_t_1).difference(set(ftr_t)))
    fscr = c/n

    return fscr


def classify(X, Y, i, selected_ftr, model, param):
    """Classify the samples of this batch

    :param numpy.ndarray X: dataset
    :param numpy.ndarray Y: target
    :param int i: current stream index (start of current batch)
    :param numpy.ndarray selected_ftr: indices of currently selected features
    :param object model: ML model
    :param dict param: parameters
    :return: model (ML model), acc(accuracy score)
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
    acc = accuracy_score(y_test, y_pred)

    return model, acc
