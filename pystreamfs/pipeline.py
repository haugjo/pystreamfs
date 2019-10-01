import numpy as np
from pystreamfs.stream_simulator import prepare_data, simulate_stream, plot_stats


class Pipeline:
    def __init__(self, dataset, generator, feature_selector, predictor, metric, param):
        self.providedData = dataset
        self.dataset = None
        self.feature_names = np.ndarray([])
        self.generator = generator
        self.feature_selector = feature_selector
        self.predictor = predictor
        self.metric = metric
        self.param = param
        self.stats = dict()

    def start(self):
        if self.providedData is not None:
            X, Y, self.feature_names = prepare_data(self.providedData, self.param['label_idx'], self.param['shuffle_data'])
            self.dataset = {'X': X, 'Y': Y}
        else:  # if generator is defined
            self.feature_names = np.arange(0, self.generator.no_features)

        self.stats = simulate_stream(self.dataset, self.generator, self.feature_selector, self.predictor, self.metric, self.param)

        return self.stats

    def plot(self):
        plot_stats(self.stats, self.feature_names, self.feature_selector.name, type(self.predictor).__name__,
                   self.metric.__name__, self.param, 0.8).show()
