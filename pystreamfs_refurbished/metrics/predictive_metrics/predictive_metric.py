from pystreamfs_refurbished.metrics.base_metric import BaseMetric
from abc import ABCMeta, abstractmethod


class PredictiveMetric(BaseMetric, metaclass=ABCMeta):
    """ Predictive Metric (Adapter)
    Get (y, y_hat) as input and measure predictive power
    """
    def __init__(self, name):
        super().__init__(name=name)

    @abstractmethod
    def compute(self, true, predicted):
        """ Compute metric given inputs at current time step and append self.measures"""
        super().compute()  # update sufficient statistics

    @staticmethod
    def sklearn_metric(metric, name):
        """Dynamically create an object of type PredictiveMetric given the specified sklearn metric"""

        # Specify compute() function
        def computations(self, true, predicted):
            self.measures.append([metric(true, predicted)])
            super().compute()  # update sufficient statistics

        # Create class and object
        ScikitMetric = type(name.capitalize(),
                            (PredictiveMetric,),
                            {'compute': computations})
        scikit_metric = ScikitMetric(name=name)

        return scikit_metric
