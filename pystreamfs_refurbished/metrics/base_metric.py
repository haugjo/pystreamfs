from abc import ABCMeta, abstractmethod
import numpy as np


class BaseMetric(metaclass=ABCMeta):
    """ Based Metric
    Base Class for any kind of metric
    >>> # Create dynamic class for scikit learn metric accuracy_score
    >>> Accuracy = type('ScikitMetric', (BaseMetric,), {'compute': lambda self, true, predicted: self.measures.append([accuracy_score(true, predicted)])})
    >>> accuracy = Accuracy()
    """
    def __init__(self):
        self.measures = []
        self.mean = None

    @abstractmethod
    def compute(self, **kwargs):
        """ Compute metric given inputs at current time step and append self.measures"""
        pass

    def get_mean(self):
        self.mean = np.mean(self.measures)
        return self.mean
