import numpy as np
from sklearn.preprocessing import MinMaxScaler


def run_efs(X, Y, param, **kw):
    """Extremal Feature Selection

    Based on a paper by Carvalho et al. 2005. This Feature Selection algorithm is based on the weights of a
    Modified Balanced Winnow classifier (as introduced in the paper).

    :param numpy.nparray X: current data batch
    :param numpy.nparray Y: labels of current batch
    :param dict param: parameters, this includes...
        numpy.ndarray u: positive model
        numpy.ndarray v: negative model
        float alpha: promotion parameter
        float beta: demotion parameter
        float threshold: threshold parameter
        float M: margin
    :return: w (feature weights), param
    :rtype numpy.ndarray, dict
    """
    
    # iterate over all elements in batch
    for x, y in zip(X, Y):

        # Convert label to -1 and 1
        y = -1 if y == 0 else 1

        # Note, the original algorithm here adds a "bias" feature that is always 1

        # Normalize x
        x = MinMaxScaler().fit_transform(x.reshape(-1, 1)).flatten()

        # Calculate score
        score = np.dot(x, param['u']) - np.dot(x, param['v']) - param['threshold']

        # If prediction was mistaken
        if score * y <= param['M']:
            # Update models for all features j
            for j, _ in enumerate(param['u']):
                if y > 0:
                    param['u'][j] = param['u'][j] * param['alpha'] * (1 + x[j])
                    param['v'][j] = param['v'][j] * param['beta'] * (1 - x[j])
                else:
                    param['u'][j] = param['u'][j] * param['beta'] * (1 - x[j])
                    param['v'][j] = param['v'][j] * param['alpha'] * (1 + x[j])

    # Compute importance score of features
    w = abs(param['u'] - param['v'])

    return w, param
