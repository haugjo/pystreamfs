import numpy as np
from pystreamfs.stream_simulator import prepare_data, simulate_stream


class Pipeline:
    def __init__(self, dataset, generator, feature_selector, visualizer, predictor, metric, param):
        self.providedData = dataset
        self.dataset = None
        self.feature_names = np.ndarray([])
        self.generator = generator
        self.feature_selector = feature_selector
        self.visualizer = visualizer
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
        self.visualizer.plot_all_stats(self).show()
