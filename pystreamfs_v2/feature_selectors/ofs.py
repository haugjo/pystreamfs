from pystreamfs_v2.feature_selectors.base_feature_selector import BaseFeatureSelector
import numpy as np
import math


class OFSFeatureSelector(BaseFeatureSelector):
    def __init__(self, n_total_ftr, n_selected_ftr):
        super().__init__('OFS', n_total_ftr, n_selected_ftr, False, False, False)

    def weight_features(self, x, y):
        """Online Feature Selection

        Based on a paper by Wang et al. 2014. Feature Selection for binary classification.
        This code is an adaptation of the official Matlab implementation.
        """
        eta = 0.2
        lamb = 0.01

        for x_b, y_b in zip(x, y):  # perform feature selection for each instance in batch
            # Convert label to -1 and 1
            y_b = -1 if y_b == 0 else 1

            f = np.dot(self.raw_weight_vector, x_b)  # prediction

            if y_b * f <= 1:  # update classifier w
                self.raw_weight_vector = self.raw_weight_vector + eta * y_b * x_b
                self.raw_weight_vector = self.raw_weight_vector * min(1, 1 / (math.sqrt(lamb) * np.linalg.norm(self.raw_weight_vector)))
                self._truncate()

    def _truncate(self):
        """Truncate the weight vector

        Set all but the **num_features** biggest absolute values to zero.
        """

        if len(self.raw_weight_vector.nonzero()[0]) > self.n_selected_ftr:
            w_sort_idx = np.argsort(abs(self.raw_weight_vector))[-self.n_selected_ftr:]
            zero_indices = [x for x in range(len(self.raw_weight_vector)) if x not in w_sort_idx]
            self.raw_weight_vector[zero_indices] = 0

    def detect_concept_drift(self, x, y):
        raise NotImplementedError
