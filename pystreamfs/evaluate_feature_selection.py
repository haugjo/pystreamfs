import time
import warnings
import copy
import numpy as np
from abc import ABCMeta

import matplotlib.pyplot as plt

from pystreamfs.metrics.time_metric import TimeMetric
from pystreamfs.utils.base_event import Event
from pystreamfs.utils.event_handlers import start_evaluation_routine, finish_iteration_routine, finish_evaluation_routine
from pystreamfs.visualization.live_plot import LivePlot


class EvaluateFeatureSelection:
    """ Main class: Evaluate the online feature selection model

    Models are processed in an interleaved test-then-train evaluation (aka. prequential evaluation)

    This class was inspired by the 'PrequentialEvaluation' class from scikit mutliflow
    (https://scikit-multiflow.readthedocs.io/en/stable/api/generated/skmultiflow.evaluation.EvaluatePrequential.html#skmultiflow.evaluation.EvaluatePrequential)

    """
    # Event handlers (class variables)
    on_start_evaluation = Event()
    on_start_evaluation.append(start_evaluation_routine)
    on_finish_iteration = Event()
    on_finish_iteration.append(finish_iteration_routine)
    on_finish_evaluation = Event()
    on_finish_evaluation.append(finish_evaluation_routine)

    def __init__(self,
                 max_samples=100000,        # (int) Maximum number of observations used in the evaluation
                 batch_size=100,            # (int) Size of one batch (i.e. no. of observations at one time step)
                 pretrain_size=100,         # (int) No. of observations used for initial training of the predictive model
                 pred_metrics=None,         # (list) Predictive metrics/measures
                 fs_metrics=None,           # (list) Feature selection metrics/measures
                 streaming_features=None,   # (dict) (time, feature index) tuples to simulate streaming features
                 output_file_path=None,     # (str) Path for the summary file output (if None, no file is created)
                 live_plot=False):          # (bool) If true, show live plot

        self.max_samples = max_samples
        self.batch_size = batch_size
        self.pretrain_size = pretrain_size
        self.pred_metrics = pred_metrics
        self.fs_metrics = fs_metrics
        self.streaming_features = dict() if streaming_features is None else streaming_features
        self.output_file_path = output_file_path
        self.live_plot = live_plot

        self.iteration = 1                  # (int) Current iteration (logical time step)
        self.start_time = 0                 # (float) Physical time when starting the evaluation
        self.global_sample_count = 0        # (int) No. of observations processed so far
        self.data_stream = None             # (skmultiflow.data.FileStream) Streaming data
        self.feature_selector = None        # (BaseFeatureSelector) Feature selection model
        self.predictor = None               # (_BasePredictiveModel) Predictive model with partial_fit() function
        self.active_features = []           # (list) Indices of currently active features (for simulating streaming features)

    def evaluate(self, data_stream, feature_selector, predictor, predictor_name=None):
        """ Evaluate the feature selection model

        :param data_stream: (skmultiflow.data.FileStream) Streaming data
        :param feature_selector: (BaseFeatureSelector) Feature selection model
        :param predictor: Predictive model with partial_fit() function (e.g. from skmultiflow or sklearn)
        :param predictor_name: (str) Name of the predictive model

        """
        self.data_stream = data_stream
        self.feature_selector = feature_selector
        self.predictor = _BasePredictiveModel(name=predictor_name, model=predictor)  # Wrap predictive model

        # Issue warning if max_samples exceeds the no. of samples available in the data stream
        if (self.data_stream.n_remaining_samples() > 0) and (self.data_stream.n_remaining_samples() < self.max_samples):
            self.max_samples = self.data_stream.n_samples
            warnings.warn('Parameter max_samples exceeds the size of data_stream and will be automatically reset.', stacklevel=2)

        # Start evaluation
        self.on_start_evaluation(self)

        # Evaluation
        if self.live_plot:  # If live visualization
            ani = LivePlot(self)
            plt.show()
        else:
            self.test_then_train()

        # Finish evaluation
        self.on_finish_evaluation(self)

    def test_then_train(self):
        """ Test-then-train evaluation """
        while self.global_sample_count < self.max_samples:
            try:
                self.one_training_iteration()
            except BaseException as exc:
                print(exc)
                break

    def one_training_iteration(self):
        # Load data batch
        if self.global_sample_count + self.batch_size <= self.max_samples:
            samples = self.batch_size
        else:
            samples = self.max_samples - self.global_sample_count  # all remaining samples
        X, y = self.data_stream.next_sample(samples)

        # Simulate streaming features
        if self.feature_selector.supports_streaming_features:
            X = self._simulate_streaming_features(X)

        # Feature Selection
        start = time.time()
        self.feature_selector.weight_features(copy.copy(X), copy.copy(y))
        self.feature_selector.comp_time.compute(start, time.time())
        self.feature_selector.select_features()
        for metric in self.fs_metrics:
            metric.compute(self.feature_selector)

        # Retain selected features
        X = self._sparsify_X(X, self.feature_selector.selection[-1])

        # Testing
        start = time.time()
        prediction = self.predictor.model.predict(X).tolist()
        self.predictor.testing_time.compute(start, time.time())
        self.predictor.predictions.append(prediction)
        for metric in self.pred_metrics:
            metric.compute(y, prediction)

        # Training
        start = time.time()
        self.predictor.model.partial_fit(X, y, self.data_stream.target_values)
        self.predictor.training_time.compute(start, time.time())

        # Finish iteration
        self.on_finish_iteration(self, samples)

    def _simulate_streaming_features(self, X):
        """ Simulate streaming features

        Remove inactive features as specified in streaming_features.

        :param X: (np.ndarray) Samples of current batch
        :return: sparse X
        :rtype np.ndarray
        """
        if self.iteration == 0 and self.iteration not in self.streaming_features:
            self.active_features = np.arange(self.feature_selector.n_total_ftr)
            warnings.warn(
                'Simulate streaming features: No active features provided at t=0. All features are used instead.')
        elif self.iteration in self.streaming_features:
            self.active_features = self.streaming_features[self.iteration]
            print('New streaming features {} at t={}'.format(self.streaming_features[self.iteration], self.iteration))

        return self._sparsify_X(X, self.active_features)

    @staticmethod
    def _sparsify_X(X, active_features):
        """ 'Remove' inactive features from X by setting them to zero

        :param X: (np.ndarray) Samples of current batch
        :param active_features: (list) Indices of active features
        :return: sparse X
        :rtype np.ndarray
        """
        sparse_X = np.zeros(X.shape)
        sparse_X[:, active_features] = X[:, active_features]
        return sparse_X


class _BasePredictiveModel(metaclass=ABCMeta):
    """ Wrapper for predictive model with partial_fit() function """

    def __init__(self, name, model):
        self.name = name                    # (str) Name of the predictive model
        self.model = model                  # Predictive model with partial_fit() function , e.g. from skmultiflow or sklearn
        self.predictions = []               # (list) Predicted label(s) per time step
        self.testing_time = TimeMetric()    # (TimeMetric) Testing times per time step
        self.training_time = TimeMetric()   # (TimeMetric) Training times per time step
