import numpy as np
from sklearn.feature_selection import mutual_info_classif


def run_mcnn(X, Y, param, **kw):
    """Feature selection based on the Micro Clusters Nearest Neighbor

    This code is based on the descriptions, formulas and pseudo code snippets from the paper by Hammodi et al.
    We cannot claim this to be the exact same implementation as intended by Hammodi et al.

    :param numpy.ndarray X: current data batch
    :param numpy.ndarray Y: labels of current data batch
    :param dict param: parameters, this includes...
        TimeWindow window: a TimeWindow object that is sequentially updated for every time window t
        dict clusters: a set of clusters
        int max_n': maximum number of saved instances per cluster
        int e_threshold: error threshold for splitting of a cluster
        max_out_of_var_bound: percentage of variables that can at most be outside of variance boundary before new cluster is created
        p_diff_threshold: threshold of perc. diff. for split/death rate when drift is assumed (_detect_drift())
    :return: w (feature weights), param (with updated window and clusters)
    :rtype numpy.ndarray, dict
    """

    # set window and clusters object
    if 'window' not in param:
        window = TimeWindow(X[0])
    else:
        window = param['window']

    if 'clusters' not in param:
        clusters = dict()
    else:
        clusters = param['clusters']

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

            # check if x percent of the dimensions are outside the boundary and create new cluster in that case
            # Note: boundary is not clearly defined in the paper. We chose this approach, because in high dimensional..
            # ..data sets we would always get outside of the boundary when looking at all dimensions
            out_of_boundary = sum(min_dist > min_c.variance)/len(min_dist)

            if out_of_boundary > param['max_out_of_var_bound']:
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

        # remove least participating cluster, but only if number of clusters is > 1
        if len(clusters) > 1:
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

    if ftr_idx.size:
        # if there are features whose ig must be updated, calculate new IG
        ig = _calc_info_gain(clusters)

    for ftr in ftr_idx:
        window = _update_info_gain(window, ig, ftr)

    # update selected features when drift detected
    if window.drift:
        w = _select_features(clusters, window)
    else:
        w = window.selected_ftr

    # update param
    param['window'] = window
    param['clusters'] = clusters

    return w, param


def _select_features(clusters, window):
    """Selection of relevant features

    After a drift is detected we want to update the relevancy of our features, i.e. find the feature which is
    responsible for drift and set consider it irrelevant for the next time window

    :param dict(MicroCluster) clusters: dictionary of currently existing Micro Clusters
    :param TimeWindow window: time window
    :return: window.selected_ftr (an updated vector of feature weights)
    :rtype TimeWindow
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


def _calc_info_gain(clusters):
    """Calculate the Information Gain of features

    :param dict(MicroCluster) clusters: dictionary of currently existing Micro Clusters
    :return: ig (information gain for all features)
    :rtype: list
    """

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

    ig = mutual_info_classif(total_data, total_labels, random_state=0)

    return ig


def _update_info_gain(window, new_ig, ftr):
    """Update the Information Gain

    Updates the Information Gain of the given feature. Called once for all features at t = 1
    and whenever there is an irrelevant feature

    :param TimeWindow window: time window
    :param dict(MicroCluster) new_ig: new information gain
    :param ftr: index of the feature for which IG shall be updated
    :return: window (updated time window)
    :rtype TimeWindow
    """

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
    """Detection of a concept drift

    Called for each time window. Checks whether a concept drift appeared in the data by looking at split and death rates

    :param TimeWindow window: time window
    :param dict param: parameters
    :return: window (updated time window)
    :rtype TimeWindow
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
    """Add and instance to a cluster

    Add the given instance x to the given cluster c.

    :param MicroCluster c: the cluster to which x shall be added
    :param int c_key: index of c in clusters
    :param numpy.ndarray x: data instance
    :param int y: label of x
    :param TimeWindow window: time window
    :param numpy.ndarray dist_sums: total distance of x to each centroid
    :param dict(MicroCluster) clusters: dictionary of currently existing Micro Clusters
    :return: clusters (clusters dictionary with updated cluster c)
    :rtype dict(MicroCluster)

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
        # increment error code
        c.e += 1

        # if false positive also increment fpr
        if y == 1:
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
    """Update cluster statistics

    Updates the statistics (centroid, n etc.) for the given cluster c.

    :param MicroCluster c: Micro Cluster that is updated
    :return: c (updated cluster)
    :rtype MicroCluster
    """

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

    Perform a cluster split of the given cluster c, because its error threshold is reached.
    The new cluster centroids are set to Q1 and Q3 of c respectively.

    :param MicroCluster c: Micro Cluster that is split
    :param TimeWindow window: time window
    :return: new_c1, new_c2 (new clusters), window (updated time window)
    :rtype MicroCluster, MicroCluster, TimeWindow
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
    """Remove a cluster

    Remove the cluster that was least recently updated if its false positive score is greater than 0.

    :param dict(MicroCluster) clusters: dictionary of currently existing Micro Clusters
    :param TimeWindow window: time window
    :return: clusters (updated clusters), window (updated time window)
    :rtype dict(MicroCluster), TimeWindow
    """
    t_diff = dict()

    # calculate the difference of time stamps for each cluster
    for key, c in clusters.items():
        t_diff[key] = window.t - np.sum(c.t)/c.n

    # find cluster with highest time difference
    max_t_c = max(t_diff, key=t_diff.get)

    # remove max_t_c if false positive rate > 0
    if clusters[max_t_c].fpr:
        clusters.pop(max_t_c, None)

        # increment death count on window
        window.deaths += 1

    return clusters, window


class _MicroCluster:
    def __init__(self, window, x, y, param):
        """Initialize new Micro Cluster

        Create a MicroCluster object every time you need a new cluster

        :param TimeWindow window: TimeWindow object
        :param numpy.ndarray x: single instance
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
        """Initialize Time Window

        Time Window object that is passed from each MCNN iteration to one another.
        You need only one TimeWindow object during MCNN feature selection.

        :param numpy.ndarray x: single (first) instance
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
