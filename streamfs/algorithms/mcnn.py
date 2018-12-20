import numpy as np
import time
import psutil
import os
from sklearn.feature_selection import mutual_info_classif


def run_mcnn(X, Y, window, clusters, param):
    """Feature selection based on the MCNN Feature Selection algorithm by Hammodi


    :return: w (updated feature weights), time (computation time in seconds),
        memory (currently used memory in percent of total physical memory)
    :rtype numpy.ndarray, float, float
    """

    start_t = time.perf_counter()  # time taking

    # update time window
    window.t += 1
    window.n = X.shape[0]
    window.split_rate_h = window.split_rate
    window.death_rate_h = window.death_rate
    window.splits = 0
    window.deaths = 0
    window.split_rate = 0
    window.death_rate = 0

    # iterate over all x in the batch
    for x, y in zip(X, Y):
        if len(clusters) != 0:
            # search for cluster that is closest to x, if their exists at least one cluster
            distances = dict()
            dist_sums = dict()

            for key, c in clusters.items():
                distances[key] = abs(c.centroid - x)  # get total distance for each cluster
                dist_sums[key] = sum(abs(c.centroid - x))

            min_c_key = min(dist_sums, key=dist_sums.get)
            min_c = clusters[min_c_key]
            min_dist = distances[min_c_key]

            # check if x is within 2x the variance boundary in all dimensions, if not create new cluster
            # Note: boundary is not clearly defined in the paper
            if (min_dist > min_c.variance * param['boundary_var_multiplier']).any():
                new_c = _MicroCluster(window, x, y, param)
                clusters[window.cluster_idx] = new_c  # add new cluster
                window.cluster_idx += 1  # increment cluster index
            else:
                # add new instance to cluster min_c and return updated clusters
                clusters = _add_instance(min_c, min_c_key, x, y, window, dist_sums, clusters)
        else:
            # create a new cluster if their doesn't exist one yet
            new_c = _MicroCluster(window, x, y, param)
            clusters[window.cluster_idx] = new_c  # add new cluster
            window.cluster_idx += 1  # increment cluster index

        # remove least participating cluster
        clusters, window = _remove_cluster(clusters, window)

    # update cluster velocity for all clusters
    for key, c in clusters.items():  # check if cluster was not removed already
            # calculate velocity of each feature
            c.velocity = abs(c.f_val / c.n - c.f_val_h / c.n_h)

            # update historic statistics
            c.f_val_h = c.f_val
            c.n_h = c.n

            clusters[key] = c

    # check for concept drift
    window = _detect_drift(window, param)

    # update information gain
    if window.t == 1:
        # initially set the info gain after the first time window
        ftr_idx = np.where(window.ftr_relevancy == 1)[0]
    else:
        # update information gain of currently irrelevant features
        ftr_idx = np.where(window.ftr_relevancy == 0)[0]

    for ftr in ftr_idx:
        window = _update_info_gain(window, clusters, ftr)

    # update selected features when drift detected
    if window.drift:
        w = _select_features(clusters, window)
    else:
        w = window.selected_ftr

    return w, window, clusters, time.perf_counter() - start_t, psutil.Process(os.getpid()).memory_percent()


def _select_features(clusters, window):
    """

    :param clusters:
    :param window:
    :return:
    """
    max_iqr_scores = np.zeros(window.selected_ftr.shape)

    # for each cluster, find the feature with highest iqr and increment its max_iqr score
    for key, c in clusters.items():
        max_iqr_idx = np.argmax(c.iqr)
        c.max_iqr[max_iqr_idx] += 1

        # add max_iqr of the cluster to total max_iqr_scores
        max_iqr_scores += c.max_iqr

    # feature with max iqr score is considered irrelevant for classification this time window
    irr_feature = np.argmax(max_iqr_scores)

    # update relevancy of newly found irrelevant feature
    window.ftr_relevancy[irr_feature] = 0

    # update selected features array
    window.selected_ftr[:] = 0
    window.selected_ftr[window.ftr_relevancy == 1] = window.ftr_ig[window.ftr_relevancy == 1]  # add weight where feature is relevant

    return window.selected_ftr


def _update_info_gain(window, clusters, ftr):
    # sum up the instances and labels of all clusters to calc. info gain
    total_data = None
    total_labels = None

    for c in clusters.values():
        # add instances of c to all data
        if total_data is None and total_labels is None:
            total_data = c.instances
            total_labels = c.instance_labels
        else:
            total_data = np.append(total_data, c.instances, axis=0)
            total_labels = np.append(total_labels, c.instance_labels)

    new_ig = mutual_info_classif(total_data, total_labels, random_state=0)

    # calculate the mean IG for this and the last time window
    mean_ig = (window.ftr_ig[ftr] + new_ig[ftr]) / 2

    # calculate percentage difference of ig
    p_diff_ig = (abs(window.ftr_ig[ftr] - new_ig[ftr]) / mean_ig) * 100

    # if percentage difference is greater than 50%, make feature relevant again
    if p_diff_ig > 50:
        window.ftr_relevancy[ftr] = 1

    # save new ig
    window.ftr_ig[ftr] = new_ig[ftr]

    # update selected features array
    window.selected_ftr[:] = 0
    window.selected_ftr[window.ftr_relevancy == 1] = window.ftr_ig[window.ftr_relevancy == 1]

    return window


def _detect_drift(window, param):
    """Detect if a concept drift appeared in this time window

    :param window:
    :return:
    """
    # calculate split and death rate
    window.split_rate = window.splits / window.n
    window.death_rate = window.deaths / window.n

    # calculate mean split and death rate for current and previous window
    mean_split_rate = (window.split_rate + window.split_rate_h) / 2
    mean_death_rate = (window.death_rate + window.death_rate_h) / 2

    # calculate percentage difference of split and death rate
    try:
        p_diff_split = (abs(window.split_rate - window.split_rate_h) / mean_split_rate) * 100
    except ZeroDivisionError:
        p_diff_split = 0

    try:
        p_diff_death = (abs(window.death_rate - window.death_rate_h) / mean_death_rate) * 100
    except ZeroDivisionError:
        p_diff_death = 0

    # if current split/death rate > mean  and percentage difference of both rates > 50%, we assume there is a drift
    split_greater_mean = window.split_rate > mean_split_rate
    death_greater_mean = window.death_rate > mean_death_rate
    p_diff_greater_50 = p_diff_split > param['p_diff_threshold'] and p_diff_death > param['p_diff_threshold']  # in the paper the threshold is 50%

    if split_greater_mean and death_greater_mean and p_diff_greater_50:
        window.drift = True
    else:
        window.drift = False

    return window


def _add_instance(c, c_key, x, y, window, dist_sums, clusters):
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
    c.instances = np.append(c.instances, [x], axis=0)
    c.t = np.append(c.t, window.t)
    c.instance_labels = np.append(c.instance_labels, int(y))
    c.n += 1

    # increment/decrement error count
    if y == c.label:
        if c.e > 0:
            c.e -= 1
    else:
        # increment error code  and fpr when x is misclassified by cluster
        c.e += 1
        c.fpr += 1

        # also increment error count of closest cluster where y = cluster.label
        dist_sums.pop(c_key, None)  # remove c from distances

        # search for next closest cluster and check if y = label
        for i in sorted(dist_sums, key=dist_sums.get):
            if clusters[i].label == y:
                clusters[i].e += 1
                break

    # check if error threshold is reached
    if c.e > c.e_threshold:
        # perform split and delete old cluster
        new_c1, new_c2, window = _split_cluster(c, window)

        clusters[window.cluster_idx] = new_c1
        window.cluster_idx += 1  # increment cluster index

        clusters[window.cluster_idx] = new_c2
        window.cluster_idx += 1  # increment cluster index

        clusters.pop(c_key, None)  # delete old cluster
    else:
        # update cluster
        clusters[c_key] = _update_cluster_stats(c)

    return clusters


def _update_cluster_stats(c):
    # delete oldest instances + its label + its time stamp until n <= max_n
    while c.n > c.max_n:
        c.instances = np.delete(c.instances, 0, 0)
        c.t = np.delete(c.t, 0, 0)
        c.instance_labels = np.delete(c.instance_labels, 0, 0)
        c.n -= 1

    c.f_val = np.sum(c.instances, axis=0)
    c.n = c.instances.shape[0]
    c.f_val2 = np.sum(c.instances ** 2, axis=0)
    c.label = np.argmax(np.bincount(c.instance_labels))
    c.variance = np.sqrt((c.f_val2 / c.n) - (c.f_val / c.n) ** 2)
    c.centroid = c.f_val / c.n

    # update q1 and q3
    c.q1 = np.percentile(c.instances, 25, axis=0)
    c.q3 = np.percentile(c.instances, 75, axis=0)
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
    for key, c in clusters.items():
        t_diff[key] = window.t - np.sum(c.t)/c.n

    # find cluster with highest time difference
    max_t_c = max(t_diff, key=t_diff.get)

    # remove max_t_c if false positive rate > 0
    if clusters[max_t_c].fpr > 0:
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

        self.f_val = np.array(x)  # summed up feature values
        self.f_val2 = np.array(x**2)  # summed up squared feature values
        self.t = np.array(window.t)  # timestamps when instances where added
        self.n = 1  # no. of instances in cluster
        self.max_n = param['max_n']  # max no. of instances in cluster
        self.label = y  # label of majority class in cluster
        self.e = 0  # error count
        self.e_threshold = param['e_threshold']  # error threshold
        self.fpr = 0  # false positive rate
        self.initial_t = window.t  # initial timestamp
        self.max_iqr = np.zeros(x.shape)  # counter for maximum iqr values

        # Todo: Hammodi et al. suggest storing the instances in a SkipList
        self.instances = np.array(x, ndmin=2)  # instances in this cluster
        self.instance_labels = np.array(int(y))

        self.centroid = self.f_val/self.n  # centroids for every feature
        self.variance = np.ones(x.shape)  # variance for every feature
        self.velocity = np.zeros(x.shape)  # velocity for every feature
        self.q1 = self.f_val  # first quartile of every feature
        self.q3 = self.f_val  # third quartile of every feature
        self.iqr = self.q3 - self.q1  # inter quartile range for each feature

        self.f_val_h = np.zeros(x.shape)  # f_val of last time window t-1
        self.n_h = 0  # n of last time window t-1


class TimeWindow:
    def __init__(self, x):
        """Initilaize Time Window

        You need only one TimeWindow object during feature selection

        :param np.array x: single (first) instance
        """

        self.t = 0  # current time step
        self.n = 0  # instances in time window = batch size
        self.cluster_idx = 0  # Index for new clusters
        self.drift = False  # indicates whether there was a drift
        self.splits = 0  # cluster splits
        self.deaths = 0  # cluster deaths
        self.split_rate = 0  # cluster split rate
        self.split_rate_h = 0  # split rate of last window
        self.death_rate = 0  # cluster death rate
        self.death_rate_h = 0  # death rate of last window
        self.ftr_relevancy = np.ones(x.shape)  # relevancy of features: 0 = irrelevant, 1 = relevant

        # Information Gain -> initially set after the first time window
        self.ftr_ig = np.ones(x.shape)

        # selected features -> initially set after the first time window
        self.selected_ftr = np.ones(x.shape)
