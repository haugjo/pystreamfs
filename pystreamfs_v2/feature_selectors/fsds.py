from pystreamfs_v2.feature_selectors.base_feature_selector import BaseFeatureSelector
import numpy as np
import numpy.linalg as ln


class FSDSFeatureSelector(BaseFeatureSelector):
    def __init__(self, n_total_ftr, n_selected_ftr, ell=0, m=None, B=None, k=2):
        super().__init__('FSDS', n_total_ftr, n_selected_ftr, False, False, False)

        if m is None:
            self.m = n_total_ftr
        else:
            self.m = m
        if B is None:
            self.B = []
        else:
            self.B = B

        self.ell = ell
        self.k = k

    def weight_features(self, x, y):
        """Feature Selection on Data Streams

            Based on a paper by Huang et al. (2015). Feature Selection for unsupervised Learning.
            This code is copied from the Python implementation of the authors with minor reductions and adaptations.
        """

        Yt = x.T  # algorithm assumes rows to represent features

        if self.ell < 1:
            self.ell = int(np.sqrt(self.m))

        if len(self.B) == 0:
            # for Y0, we need to first create an initial sketched matrix
            self.B = Yt[:, :self.ell]
            C = np.hstack((self.B, Yt[:, self.ell:]))
            n = Yt.shape[1] - self.ell
        else:
            # combine current sketched matrix with input at time t
            # C: m-by-(n+ell) matrix
            C = np.hstack((self.B, Yt))
            n = Yt.shape[1]

        U, s, V = ln.svd(C, full_matrices=False)
        U = U[:, :self.ell]
        s = s[:self.ell]
        V = V[:, :self.ell]

        # shrink step in Frequent Directions algorithm
        # (shrink singular values based on the squared smallest singular value)
        delta = s[-1] ** 2
        s = np.sqrt(s ** 2 - delta)

        # -- Extension of original code --
        # replace nan values with 0 to prevent division by zero error for small batch numbers
        s = np.nan_to_num(s)

        # update sketched matrix B
        # (focus on column singular vectors)
        self.B = np.dot(U, np.diag(s))

        # According to Section 5.1, for all experiments,
        # the authors set alpha = 2^3 * sigma_k based on the pre-experiment
        alpha = (2 ** 3) * s[self.k - 1]

        # solve the ridge regression by using the top-k singular values
        # X: m-by-k matrix (k <= ell)
        D = np.diag(s[:self.k] / (s[:self.k] ** 2 + alpha))

        # -- Extension of original code --
        # replace nan values with 0 to prevent division by zero error for small batch numbers
        D = np.nan_to_num(D)

        X = np.dot(U[:, :self.k], D)

        self.raw_weight_vector = np.amax(abs(X), axis=1)

    def detect_concept_drift(self, x, y):
        raise NotImplementedError
