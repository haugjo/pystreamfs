from abc import ABCMeta, abstractmethod
import warnings
import numpy as np

from pystreamfs.metrics.time_metric import TimeMetric


class BaseFeatureSelector(metaclass=ABCMeta):
    """ Abstract super class for all online feature selection algorithms """
    def __init__(self,
                 name,                                  # (str) Name of FS model
                 n_total_ftr,                           # (int) Total no. of features
                 n_selected_ftr,                        # (int) No. of selected features
                 supports_multi_class=False,            # (bool) True if model supports multi class classification
                 supports_streaming_features=False):    # (bool) True if model supports streaming features

        self.name = name
        self.n_total_ftr = n_total_ftr
        self.n_selected_ftr = n_selected_ftr
        self.supports_multi_class = supports_multi_class
        self.supports_streaming_features = supports_streaming_features

        self.raw_weight_vector = np.zeros(self.n_total_ftr)                         # (np.ndarray) Current weights (as produced by FS model)
        self.weights = []                                                           # (list) Absolute weights in all time steps
        self.selection = []                                                         # (list) Indices of selected features in all time steps
        self.comp_time = TimeMetric()                                               # (list) Computation times in all time steps

        # Private attributes
        self._auto_scale = False                                                    # (bool) Indicator for scaling of weights

    @abstractmethod
    def weight_features(self, X, y):
        """ Weight features

        :param X: (np.ndarray) Samples of current batch
        :param y: (np.ndarray) Labels of current batch

        """
        pass

    def select_features(self):
        """ Select features with highest absolute weights """

        if np.any(self.raw_weight_vector < 0):  # If vector contains negative weights -> issue warning
            abs_weights = abs(self.raw_weight_vector)
            if not self._auto_scale:
                warnings.warn('Weight vector contains negative weights. Absolute weights will be used for feature selection.')
                self._auto_scale = True
        else:
            abs_weights = self.raw_weight_vector

        self.weights.append(abs_weights.tolist())
        self.selection.append(np.argsort(abs_weights)[::-1][:self.n_selected_ftr].tolist())
