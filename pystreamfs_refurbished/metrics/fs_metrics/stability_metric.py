from pystreamfs_refurbished.metrics.fs_metrics.fs_metric import FSMetric
import numpy as np


class NogueiraStabilityMetric(FSMetric):
    def __init__(self, sliding_window):
        super().__init__(name='FS Stability')
        self.sliding_window = sliding_window

    def compute(self, fs_model):  # Todo: check if this still works
        # Construct Z matrix: indicates which features were selected at each time step
        Z = np.zeros([min(len(fs_model.selection), self.sliding_window), fs_model.n_total_ftr])
        for row, col in enumerate(fs_model.selection[-self.sliding_window:]):
            Z[row, col] = 1

        ''' START ORIGINAL CODE
        Let us assume we have M>1 feature sets and d>0 features in total.
        This function computes the stability estimate as given in Definition 4 in  [1].

        INPUT: A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d).
               Each row of the binary matrix represents a feature set, where a 1 at the f^th position
               means the f^th feature has been selected and a 0 means it has not been selected.

        OUTPUT: The stability of the feature selection procedure
        '''
        try:
            M, d = Z.shape
            hatPF = np.mean(Z, axis=0)
            kbar = np.sum(hatPF)
            denom = (kbar / d) * (1 - kbar / d)
            stability_measure = 1 - (M / (M - 1)) * np.mean(np.multiply(hatPF, 1 - hatPF)) / denom

            '''END ORIGINAL CODE'''
        except ZeroDivisionError:
            stability_measure = 0  # metric requires at least 2 measurements and thus runs an error at t=1

        self.measures.extend([stability_measure])
        super().compute(None)  # update sufficient statistics
