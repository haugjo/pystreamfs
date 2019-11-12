from abc import ABCMeta, abstractmethod
import numpy as np


class BaseMetric(metaclass=ABCMeta):
    """ Base Metric
    Base Class for any kind of metric
    """
    def __init__(self, name):
        self.measures = []
        self.name = name
        self.mean = None
        self.var = None

    @abstractmethod
    def compute(self, **kwargs):
        """ Compute metric given inputs at current time step and append self.measures
        + compute current mean and variance
        """
        self.mean = np.mean(self.measures)
        self.var = np.var(self.measures)
