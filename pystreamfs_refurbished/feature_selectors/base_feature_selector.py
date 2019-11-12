from abc import ABCMeta, abstractmethod
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from pystreamfs_refurbished.metrics.time_metric import TimeMetric


class BaseFeatureSelector(metaclass=ABCMeta):
    """ Abstract super class for Online Feature Selection Algorithm

    Inherit from this class to implement your own feature selection algorithm

    """
    def __init__(self, name, n_total_ftr, n_selected_ftr, supports_multi_class=False, supports_streaming_features=False, supports_concept_drift_detection=False):
        self.name = name
        self.n_total_ftr = n_total_ftr
        self.n_selected_ftr = n_selected_ftr
        self.raw_weight_vector = np.zeros(self.n_total_ftr)  # holds current weights (unscaled, as produced by feature selector)
        self.weights = []
        self.selection = []
        self.concept_drifts = []
        self.comp_time = TimeMetric()
        self.supports_multi_class = supports_multi_class
        self.supports_streaming_features = supports_streaming_features
        self.supports_concept_drift_detection = supports_concept_drift_detection

        self._auto_scale = False  # indicates if weights are scaled automatically before selection

    @abstractmethod
    def weight_features(self, x, y):
        """ Weight features
        x: samples of current batch
        y: labels of current batch
        """
        pass

    def select_features(self):
        """Select features with highest absolute weights"""

        # Check if feature weights are normalized in range [0,1]
        if np.any((self.raw_weight_vector < 0) | (self.raw_weight_vector > 1)):
            scaled_weights = MinMaxScaler().fit_transform(self.raw_weight_vector.reshape(-1, 1)).flatten()
            if not self._auto_scale:
                warnings.warn('Feature weights are automatically scaled to range [0,1] before selection.')
                self._auto_scale = True
        else:
            scaled_weights = self.raw_weight_vector

        self.weights.append(scaled_weights)
        self.selection.append(np.argsort(scaled_weights)[::-1][:self.n_selected_ftr])

    @abstractmethod
    def detect_concept_drift(self, x, y):
        """ Check for concept drift at current time step"""
        pass
