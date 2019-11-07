from pystreamfs_refurbished.metrics.base_metric import BaseMetric
from abc import ABCMeta, abstractmethod


class PredictiveMetric(BaseMetric, metaclass=ABCMeta):
    """ Predictive Metric (Adapter)
    Get (y, y_hat) as input and measure predictive power
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute(self, true, predicted):
        """ Compute metric given inputs at current time step and append self.measures"""
        pass
