import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def comp_fscr(ftr_t_1, ftr_t, n):
    c = len(set(ftr_t_1).difference(set(ftr_t)))

    # we do not need to divide by 2n, because c only counts where a new feature is selected and not the ones unselected
    fscr = c/n

    return fscr


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
        model = KNeighborsClassifier(n_jobs=-1)
    elif model is None and param['algorithm'] == "tree":
        model = DecisionTreeClassifier(random_state=0)
    elif model is None and param['algorithm'] == "svm":
        model = SVC()

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
