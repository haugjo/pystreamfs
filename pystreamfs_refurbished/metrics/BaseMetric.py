from abc import ABCMeta, abstractmethod


class BaseMetric(metaclass=ABCMeta):
    """ Based Metric (Adapter)
    Inherit this class to implement custom metric
    OR
    refer to pre-defined scikit-learn metric:
    >>> # Create dynamic class for scikit learn metric accuracy_score
    >>> Accuracy = type('ScikitMetric', (BaseMetric,), {'compute': lambda true, predicted: sklearn.metrics.accuracy_score(true, predicted)})
    >>> accuracy = Accuracy()
    """
    def __init__(self):
        self.measures = []

    @abstractmethod
    def compute(self):
        """ Compute metric given inputs at current time step and append self.measures"""
        pass
