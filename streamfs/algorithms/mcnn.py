import numpy as np
import time
import psutil
import os
import math


def run_mcnn(x, y, window, clusters, param):
    """Python adaptation of the MCNN Feature Selection algorithm by Hammodi


    :return: w (updated feature weights), time (computation time in seconds),
        memory (currently used memory in percent of total physical memory)
    :rtype numpy.ndarray, float, float
    """

    start_t = time.process_time()  # time taking

    # search for cluster that is closest to x
    distances = dict()

    for c in clusters:
        distances[c] = abs(clusters[c].centroid - x)

    min_c_key = min(distances, key=distances.get)
    min_c = clusters[min_c_key].copy()  # get copy of cluster
    min_dist = min(distances)

    # check if x is within the variance boundary of that cluster, if not create new cluster
    if min_dist > min_c.variance:
        new_c = _MicroCluster(window, x, y, param)
        clusters[len(clusters)] = new_c  # add new cluster
    else:
        # add new instance to cluster min_c and return updated clusters
        clusters = _add_instance(min_c, min_c_key, x, y, window, distances, clusters)

    return w, clusters, time.process_time() - start_t, psutil.Process(os.getpid()).memory_percent()


def _add_instance(c, c_key, x, y, window, distances, clusters):
    """Add the given instance to the given cluster

    :param c:
    :param c_key:
    :param x:
    :param y:
    :param window:
    :param distances:
    :param clusters:
    :return:

    ...todo... authors suggest to apply Lowpass Filter to instance and cluster before adding it
    """
    # add x and its timestamp to the cluster
    c.instances = np.append(c.instances, x, axis=0)
    c.t = np.append(c.t, window.t)

    # increment/decrement error count
    if y == c.label:
        c.e -= 1
    else:
        c.e += 1  # increment error code when x is misclassified by cluster

        # also increment error count of closest cluster where y = cluster.label
        distances.pop(c_key, None)  # remove c from distances

        # search for next closest cluster and check if y = label
        for i in sorted(distances, key=distances.get):
            if clusters[i].label == y:
                clusters[i].e += 1
                break

    # check if error threshold is reached
    if c.e > c.e_threshold:
        # perform split and delete old cluster
        new_c1, new_c2, window = _split_cluster(c, window)

        clusters[len(clusters)] = new_c1
        clusters[len(clusters)] = new_c2
        clusters.pop(c_key, None)
    else:
        clusters[c_key] = c  # update cluster

    return clusters


def _split_cluster(c, window):
    """Split a cluster

    :param c:
    :param window:
    :return:
    """

    param = dict()
    param['max_n'] = c.max_n
    param['e_threshold'] = c.e_threshold

    new_c1 = _MicroCluster(window, c.q1, c.label, param)
    new_c2 = _MicroCluster(window, c.q3, c.label, param)

    # increment split count of window
    window.splits += 1

    return new_c1, new_c2, window


class _MicroCluster:
    def __init__(self, window, x, y, param):
        """Initialize new Micro Cluster

        Create a MicroCluster object every time you need a new cluster (when splitting old clusters)

        :param TimeWindow window: TimeWindow object
        :param np.array x: single instance
        :param int y: class label of x
        :param dict param: parameters
        """

        self.f_val = x  # summed up feature values
        self.f_val2 = x**2  # summed up squared feature values
        self.t = window.t  # timestamps when instances where added
        self.n = 1  # no. of instances in cluster
        self.max_n = param['max_n']  # max no. of instances in cluster
        self.label = y  # label of majority class in cluster
        self.e = 0  # error count
        self.e_threshold = param['e_threshold']  # error threshold
        self.initial_t = window.t  # initial timestamp
        self.max_iqr = np.zeros(x.shape)  # counter for maximum iqr values

        # Todo: Hammodi et al. suggest storing the instances in a SkipList
        self.instances = x  # instances in this cluster

        self.centroid = self.f_val/self.n  # centroids for every feature
        self.variance = math.sqrt((self.f_val2/self.n)-(self.f_val/self.n)**2)  # variance for every feature
        self.velocity = np.zeros(x.shape)  # velocity for every feature
        self.q1 = np.percentile(self.f_val, 25, axis=1)  # first quartile of every feature
        self.q3 = np.percentile(self.f_val, 75, axis=1)  # third quartile of every feature

        self.f_val_h = np.zeros(x.shape)  # f_val of last time window t-1
        self.n_h = 0  # n of last time window t-1


class _TimeWindow:
    def __init__(self, x):
        """Initilaize Time Window

        You need only one TimeWindow object during feature selection

        :param np.array x: single (first) instance
        """

        self.t = 0  # current time step
        self.n = 0  # instances in time window = batch size
        self.drift = False  # indicates whether there was a drift
        self.splits = 0  # cluster splits
        self.deaths = 0  # cluster deaths
        self.split_rate = 0  # cluster split rate
        self.death_rate = 0  # cluster death rate
        self.p_diff_split = 0  # Percentage Difference of split rate
        self.p_diff_death = 0  # Percentage Difference of death rate
        self.irr_ftr = np.zeros(x.shape)  # irrelevant features of last time window
        self.rel_ftr = np.zeros(x.shape)  # relevant features
        self.ig_ftr = np.zeros(x.shape)  # Information Gain for every feature




