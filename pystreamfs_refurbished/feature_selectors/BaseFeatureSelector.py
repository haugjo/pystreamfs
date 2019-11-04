from abc import ABCMeta, abstractmethod


class BaseFeatureSelector(metaclass=ABCMeta):
    """ Abstract super class for Online Feature Selection Algorithm

    Inherit from this class to implement your own feature selection algorithm for use in pystreamfs

    """
    def __init__(self):
        self.n_features = 0
        self.supports_streaming_features = False
        self.supports_concept_drift_detection = False

    @abstractmethod
    def select_features(self, x, y):
        """ Perform feature selection
        x: samples of current batch
        y: labels of current batch
        :return: selected features
        """
        pass
