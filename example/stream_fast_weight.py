# coding: utf-8

"""
Implementation of algorithms proposed by:

    H. Huang, et al., "Unsupervised Feature Selection on Data Streams," Proc. of CIKM 2015, pp. 1031-1040 (Oct. 2015).
"""

import numpy as np
import numpy.linalg as ln

class StreamFastWeight:
    """
    Alg. 2: Streaming update of feature weights at time t
    """

    def __init__(self, m, k, ell=0):
        """
        :param m: number of original features
        :param k: number of singular vectors (this can be the same as the number of clusters in the dataset)
        :param ell: sketche size for a sketched m-by-ell matrix B
        """

        self.m = m
        self.k = k
        if ell < 1: self.ell = int(np.sqrt(self.m))
        else: self.ell = ell

    def update(self, Yt):
        """
        Update the sketched matrix B based on new inputs at time t,
        and return weight of each feature
        :param Yt: m-by-n_t input matrix from data stream
        """

        # combine current sketched matrix with input at time t
        # C: m-by-(n+ell) matrix
        if hasattr(self, 'B'):
            C = np.hstack((self.B, Yt))
            n = Yt.shape[1]
        else:
            # for Y0, we need to first create an initial sketched matrix
            self.B = Yt[:, :self.ell]
            C = np.hstack((self.B, Yt[:, self.ell:]))
            n = Yt.shape[1] - self.ell

        U, s, V = ln.svd(C, full_matrices=False)
        U = U[:, :self.ell]
        s = s[:self.ell]
        V = V[:, :self.ell]

        # shrink step in Frequent Directions algorithm
        # (shrink singular values based on the squared smallest singular value)
        delta = s[-1] ** 2
        s = np.sqrt(s ** 2 - delta)

        # update sketched matrix B
        # (focus on column singular vectors)
        self.B = np.dot(U, np.diag(s))

        # According to Section 5.1, for all experiments,
        # the authors set alpha = 2^3 * sigma_k based on the pre-experiment
        alpha = (2 ** 3) * s[self.k-1]

        # solve the ridge regression by using the top-k singular values
        # X: m-by-k matrix (k <= ell)
        D = np.diag(s[:self.k] / (s[:self.k] ** 2 + alpha))
        X = np.dot(U[:, :self.k], D)

        return np.amax(abs(X), axis=1)
