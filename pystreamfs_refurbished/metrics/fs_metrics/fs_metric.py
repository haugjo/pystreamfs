from pystreamfs_refurbished.metrics.base_metric import BaseMetric
from abc import ABCMeta, abstractmethod


class FSMetric(BaseMetric, metaclass=ABCMeta):
    """ Feature Selection Metric (Adapter)
    Gets feature selection model as input and computes related metric
    """
    def __init__(self, name):
        super().__init__(name=name)

    @abstractmethod
    def compute(self, fs_model):
        """ Compute metric given inputs at current time step and append self.measures"""
        super().compute()  # update sufficient statistics
