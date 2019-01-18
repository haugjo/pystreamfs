import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def truncate(w, num_features):
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


def comp_mfcr(ftr_t_1, ftr_t, total_no_ftr, t, mfcr):
    # number of changes with regard to t-1
    # every change in the selected feature set indicates 2 changes of feature weights -> thus we multiply by 2
    c = len(set(ftr_t_1).difference(set(ftr_t))) * 2

    if t == 1:
        return c / (total_no_ftr * t)
    else:
        return mfcr * (t-1)/t + c / (total_no_ftr * t)


def perform_learning(X, y, i, selected_ftr, model, param):
    """

    :param numpy.ndarray X: dataset
    :param numpy.ndarray Y: target
    :param int i: current stream index (start of current batch)
    :param numpy.ndarray selected_ftr: indices of currently selected features
    :param object model: ML model (either KNN or Decision Tree classifier
    :param dict param: parameters for feature selection

    :return: model (ML model), acc(accuracy score)
    :rtype: object, float
    """

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
    if type(model) is KNeighborsClassifier:
        if X_train.shape[0] < param['neighbors']:
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
