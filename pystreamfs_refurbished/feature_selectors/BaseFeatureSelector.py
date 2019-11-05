from abc import ABCMeta, abstractmethod
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class BaseFeatureSelector(metaclass=ABCMeta):
    """ Abstract super class for Online Feature Selection Algorithm

    Inherit from this class to implement your own feature selection algorithm

    """
    def __init__(self, n_total_ftr, n_selected_ftr, supports_multi_class=False, supports_streaming_features=False, supports_concept_drift_detection=False):
        self.n_total_ftr = n_total_ftr
        self.n_selected_ftr = n_selected_ftr
        self.weights = np.zeros(n_total_ftr)
        self.selection = None
        self.concept_drifts = None
        self.supports_multi_class = supports_multi_class
        self.supports_streaming_features = supports_streaming_features
        self.supports_concept_drift_detection = supports_concept_drift_detection

    @abstractmethod
    def weight_features(self, x, y, active_features):
        """ Weight features
        x: samples of current batch
        y: labels of current batch
        """
        pass

    def select_features(self):
        """Select features with highest absolute weights"""

        # Check if feature weights are normalized in range [0,1]
        if np.any(self.weights < 0 | self.weights > 1):
            scaled_weights = MinMaxScaler().fit_transform(self.weights.reshape(-1, 1)).flatten()
            warnings.warn('Feature weights were automatically scaled to range [0,1] before selection.')
        else:
            scaled_weights = self.weights

        self.selection = np.argsort(abs(scaled_weights))[::-1][:self.n_selected_ftr]

    @abstractmethod
    def detect_concept_drift(self):
        """ Check for concept drift at current time step"""
        pass
