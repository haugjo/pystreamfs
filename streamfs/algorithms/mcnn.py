import numpy as np
import time
import psutil
import os
import math
from sklearn.feature_selection import mutual_info_classif


def run_mcnn(X, Y, window, clusters, param):
    """Python adaptation of the MCNN Feature Selection algorithm by Hammodi


    :return: w (updated feature weights), time (computation time in seconds),
        memory (currently used memory in percent of total physical memory)
    :rtype numpy.ndarray, float, float
    """

    start_t = time.process_time()  # time taking

    # update time window
    window.t += 1
    window.n = X.shape[0]
    window.splits_h = window.splits  # save statistics from last time window
    window.deaths_h = window.deaths
    window.split_rate_h = window.split_rate
    window.death_rate_h = window.death_rate
    window.splits = 0
    window.deaths = 0
    window.split_rate = 0
    window.death_rate = 0

    updated_clusters = []

    # iterate over all x in the batch
    for x, y in zip(X, Y):
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
            updated_clusters.append(min_c_key)  # save cluster for later updating of its statistics

    # remove least participating cluster
    clusters, window = _remove_cluster(clusters, window)

    # update cluster statistics
    for c_key in updated_clusters:
        clusters[c_key] = _update_cluster_stats(clusters[c_key])

    # check for concept drift
    window = _detect_drift(window)

    # update selected features when drift detected
    if window.drift:
        w = _select_features(clusters, window, param['num_features'])
    else:
        w = window.selected_ftr

    return w, window, clusters, time.process_time() - start_t, psutil.Process(os.getpid()).memory_percent()


def _select_features(clusters, window, num_features):
    """

    :param clusters:
    :param window:
    :return:
    """
    max_iqr_scores = np.zeros(clusters[0].instances[0].shape)
    total_data = []
    total_labels = []

    # for each cluster, find the feature with highest iqr and increment its max_iqr score
    for c in clusters:
        max_iqr_idx = max(c.iqr, key=c.iqr.get)
        c.max_iqr[max_iqr_idx] += 1

        # add max_iqr of the cluster to total max_iqr_scores
        max_iqr_scores += c.max_iqr

        # add instances of c to all data
        total_data = np.append(total_data, c.instances, axis=0)
        total_labels = np.append(total_labels, c.instance_labels, axis=0)

    # feature with max iqr score is considered irrelevant for classification this time window
    irr_feature = max(max_iqr_scores, key=max_iqr_scores.get)

    # update the information gain for all irrelevant features of last time window
    irr_ftr_idx = window.ftr_relevancy[window.ftr_relevancy == 0]

    for ftr in irr_ftr_idx:
        new_ig = mutual_info_classif(total_data[ftr], total_labels, random_state=0)

        # calculate the mean IG for this and the last time window
        mean_ig = (window.ftr_ig[ftr] + new_ig) / 2

        # calculate percentage difference of ig
        p_diff_ig = (abs(window.ftr_ig[ftr] - new_ig)/mean_ig) * 100

        # if percentage difference is greater than 50%, make feature relevant again
        if p_diff_ig > 50:
            window.ftr_relevancy[ftr] = 1

        # save new ig
        window.ftr_ig[ftr] = new_ig

    # update relevancy of newly found irrelevant feature
    window.ftr_relevancy[irr_feature] = 0

    # select top features
    window.selected_ftr = np.argsort(window.ftr_ig[window.ftr_relevancy == 1])[:num_features]

    return window.selected_ftr


def _detect_drift(window):
    """Detect if a concept drift appeared in this time window

    :param window:
    :return:
    """
    # calculate split and death rate
    window.split_rate = (window.splits - window.splits_h)/window.n
    window.death_rate = (window.deaths - window.detaths_h) / window.n

    # calculate mean split and death rate for current and previous window
    mean_split_rate = (window.split_rate + window.split_rate_h) / 2
    mean_death_rate = (window.death_rate + window.death_rate_h) / 2

    # calculate percentage difference of split and death rate
    p_diff_split = (abs(window.split_rate - window.split_rate_h)) / ((window.split_rate + window.split_rate_h) / 2) * 100
    p_diff_death = (abs(window.death_rate - window.death_rate_h)) / ((window.death_rate + window.death_rate_h) / 2) * 100

    # if current split/death rate > mean  and percentage difference of both rates > 50%, we assume there is a drift
    split_greater_mean = window.split_rate > mean_split_rate
    death_greater_mean = window.death_rate > mean_death_rate
    p_diff_greater_50 = p_diff_split > 50 and p_diff_death > 50

    if split_greater_mean and death_greater_mean and p_diff_greater_50:
        window.drift = True
    else:
        window.drift = False

    return window


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
    c.instance_labels = np.append(c.instance_labels, y)

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
        # update cluster
        clusters[c_key] = c

    return clusters


def _update_cluster_stats(c):
    # delete oldest instances + its label + its time stamp until n <= max_n
    while c.n > c.max_n:
        np.delete(c.instances, 0, 0)
        np.delete(c.t, 0, 0)
        np.delete(c.instance_labels, 0, 0)

    # update basic statistics
    c.f_val_h = c.f_val
    c.n_h = c.n
    c.f_val = np.sum(c.instances, axis=1)
    c.n = c.instances.shape[0]
    c.f_val2 = np.sum(c.instances ** 2, axis=1)
    c.label = np.argmax(np.bincount(c.instance_labels))
    c.variance = math.sqrt((c.f_val2/c.n)-(c.f_val/c.n)**2)
    c.centroid = c.f_val/c.n

    # calculate velocity of each feature
    c.velocity = abs(c.f_val/c.n - c.f_val_h/c.n_h)

    # update q1 and q3
    c.q1 = np.percentile(c.f_val, 25, axis=1)
    c.q3 = np.percentile(c.f_val, 75, axis=1)
    c.iqr = c.q3 - c.q1

    return c


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


def _remove_cluster(clusters, window):
    """Remove cluster that was least recently updated

    :param clusters:
    :param window:
    :return:
    """
    t_diff = dict()

    # calculate the difference of time stamps for each cluster
    for c in clusters:
        t_diff[c] = window.t - sum(c.t)/c.n

    # find cluster with highest time difference
    max_t_c = max(t_diff, key=t_diff.get)

    # remove max_t_c if error > 0
    if clusters[max_t_c].e > 0:
        clusters.pop(max_t_c, None)

    # increment death count on window
    window.deaths += 1

    return clusters, window


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
        self.instance_labels = y

        self.centroid = self.f_val/self.n  # centroids for every feature
        self.variance = math.sqrt((self.f_val2/self.n)-(self.f_val/self.n)**2)  # variance for every feature
        self.velocity = np.zeros(x.shape)  # velocity for every feature
        self.q1 = np.percentile(self.f_val, 25, axis=1)  # first quartile of every feature
        self.q3 = np.percentile(self.f_val, 75, axis=1)  # third quartile of every feature
        self.iqr = self.q3 - self.q1  # inter quartile range for each feature

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
        self.splits_h = 0  # splits of last window
        self.deaths = 0  # cluster deaths
        self.deaths_h = 0  # deaths of last window
        self.split_rate = 0  # cluster split rate
        self.split_rate_h = 0  # split rate of last window
        self.death_rate = 0  # cluster death rate
        self.death_rate_h = 0  # death rate of last window
        self.ftr_relevancy = np.ones(x.shape)  # relevancy of features: 0 = irrelevant, 1 = relevant
        self.selected_ftr = []  # selected features (based on num_features param)
        self.ftr_ig = np.zeros(x.shape)  # Information Gain for every feature




