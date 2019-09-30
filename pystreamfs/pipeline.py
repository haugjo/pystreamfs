import numpy as np
from pystreamfs.stream_simulator import prepare_data, simulate_stream, plot_stats


class Pipeline:
    def __init__(self, dataset, generator, feature_selector, predictor, metric, param):
        self.realData = dataset
        self.features = np.ndarray([])
        self.generator = generator
        self.fs_algorithm = feature_selector
        self.predictor = predictor
        self.metric = metric
        self.param = param
        self.stats = dict()

    def start(self):
        X, Y, self.features = prepare_data(self.realData, self.param['label_idx'], self.param['shuffle_data'])
        self.stats = simulate_stream(X, Y, self.fs_algorithm, self.predictor, self.metric, self.param)

        return self.stats

    def plot(self):
        plot_stats(self.stats, self.features, self.fs_algorithm.name, type(self.predictor).__name__,
                   self.metric.__name__, self.param, 0.8).show()
