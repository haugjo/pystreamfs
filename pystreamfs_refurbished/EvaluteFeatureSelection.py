import os
import re
import sys
from timeit import default_timer as timer
import numpy as np

from skmultiflow.utils import constants
from skmultiflow.data.base_stream import Stream
from skmultiflow.evaluation.base_evaluator import StreamEvaluator

from pystreamfs_refurbished.metrics import TimeMetric
from pystreamfs_refurbished.exceptions import InvalidModelError
from pystreamfs_refurbished.feature_selectors import BaseFeatureSelector


class EvaluateFeatureSelection:
    """ Evaluation of online Feature Selection algorithms
    using the prequential evaluation method or interleaved test-then-train method.

    Reduced version of PrequentialEvaluation from scikit-multiflow, optimized for online feature selection
    """

    def __init__(self,
                 n_wait=200,
                 max_samples=100000,
                 batch_size=1,
                 pretrain_size=200,
                 max_time=float("inf"),
                 predictive_metric=None,
                 fs_metric=None,
                 streaming_features=None,
                 output_file=None,
                 show_plot=True,
                 show_live_plot=False,
                 restart_stream=True):

        # Scikit-multiflow parameters
        # self.n_sliding = n_wait  # Todo: consider using sliding window for performance measurement
        self.max_samples = max_samples  # max samples to draw from stream
        self.pretrain_size = pretrain_size  # number of samples to pretrain model at t=0
        self.batch_size = batch_size  # size of data batch at time t
        self.max_time = max_time  # max time to run the experiment
        self.output_file = output_file  # path of the output file
        self.restart_stream = restart_stream  # restarting stream at end of simulation
        self.stream = None

        # Prediction related parameters
        self.predictive_model = None
        self.predictive_model_name = None
        self.predictions = None

        # Visualization related parameters
        self.show_plot = show_plot
        self.show_live_plot = show_live_plot

        # Feature Selection related parameters
        self.fs_model = None
        self.fs_model_name = None
        self.feature_weights = None  # feature weights for every time t Todo: Store in Feature Selector
        self.selected_features = None  # selected features for every time t
        self.streaming_features = streaming_features  # time/feature-pairs for simulation of streaming features
        self.active_features = None  # currently active features

        # Metric related parameters
        self.metrics = {'predictive': predictive_metric,
                        'training_time': TimeMetric,
                        'testing_time': TimeMetric,
                        'fs': fs_metric,
                        'fs_time': TimeMetric}

        # Further general parameters
        self.time = 1  # logical time/iteration of data stream
        self.global_sample_count = 0  # count all seen samples
        self.results = dict()  # storage for all results and metrics
        self._start_time = 0  # global start time

    def evaluate(self, stream, fs_model, predictive_model, fs_model_name=None, predictive_model_name=None):
        """ Evaluate a feature selection algorithm
        In future: compare multiple feature selection algorithms
        """
        self._start_time = timer()  # start experiment

        self.stream = stream
        self.fs_model = fs_model
        self.predictive_model = predictive_model
        self.fs_model_name = fs_model_name
        self.predictive_model_name = predictive_model_name

        # check if specified models are valid
        self.__check_configuration()

        if self.pretrain_size > 0:
            self.__pretrain_predictive_model()  # pretrain model at time t=0

        # Simulate stream with feature selection using prequential evaluation
        self.__train_and_test()

    def __check_configuration(self):
        if not isinstance(self.stream, Stream):
            raise InvalidModelError('Specified data stream is not of type Stream (scikit-multiflow data type)')
        if not isinstance(self.fs_model, BaseFeatureSelector):
            raise InvalidModelError('Specified feature selection model is not of type BaseFeatureSelector '
                                    '(pystreamfs data type)')
        if not isinstance(self.predictive_model, StreamEvaluator):
            raise InvalidModelError('Specified predictive model is not of type StreamEvaluator '
                                    '(scikit-multiflow data type)')

    def __pretrain_predictive_model(self):
        print('Pre-training on {} sample(s).'.format(self.pretrain_size))  # Todo: what if no pretraining???

        x, y = self.stream.next_sample(self.pretrain_size)

        # Prediction WITHOUT feature selection and computation of metrics
        self.predictive_model.partial_fit(X=x, y=y, classes=self.stream.target_values)

        # Increase global sample count
        self.global_sample_count += self.pretrain_size

    def __train_and_test(self):
        """ Prequential evaluation """
        print('Prequential Evaluation')
        print('Evaluating {} target(s).'.format(self.stream.n_targets))

        print('Evaluating...')
        while ((self.global_sample_count < self.max_samples) & (timer() - self._start_time < self.max_time)
               & (self.stream.has_more_samples())):
            try:
                # Load batch
                x, y = self.stream.next_sample(self.batch_size)

                if x is not None and y is not None:
                    # Streaming Features
                    if self.time in self.streaming_features:
                        self.active_features = self.streaming_features[self.time]
                    elif self.active_features is None:
                        self.active_features = np.arange(x[1])  # all features are active

                    # Feature Selection
                    start = timer()
                    self.fs_model.weight_features(x, y, self.active_features)
                    self.metrics['fs_time'].compute(start, timer())
                    self.fs_model.select_features()
                    self.metrics['fs'].compute(self.selected_features, self.fs_model.n_total_ftr)
                    # Todo: store in feature selection object
                    self.feature_weights.append(self.fs_model.weights)
                    self.selected_features.append(self.fs_model.selection)

                    # Sparsify batch
                    sparse_matrix = np.zeros(x.shape())
                    sparse_matrix[:, self.fs_model.selection] = x
                    x = sparse_matrix

                    # Testing
                    start = timer()
                    prediction = self.predictive_model.predict(x)
                    self.predictions.append(prediction)
                    self.metrics['prediction'].compute(y, prediction)  # Todo: How to generalize??
                    self.metrics['testing_time'].compute(start, timer())

                    # Training
                    start = timer()
                    self.model.partial_fit(x, y, self.stream.target_values)
                    self.metrics['training_time'].compute(start, timer())

                    # Update global sample count and time
                    self.time += 1
                    self.global_sample_count += self.batch_size

            except BaseException as exc:
                print(exc)
                break

        # Todo: continue here!

        # Flush file buffer, in case it contains data
        self._flush_file_buffer()

        if len(set(self.metrics).difference({constants.DATA_POINTS})) > 0:
            self.evaluation_summary()
        else:
            print('Done')

        if self.restart_stream:
            self.stream.restart()

        return self.model

    def get_info(self):  # Todo: what about this?
        info = self.__repr__()
        if self.output_file is not None:
            _, filename = os.path.split(self.output_file)
            info = re.sub(r"output_file=(.\S+),", "output_file='{}',".format(filename), info)

        return info

    def update_progress_bar(curr, total, steps, time):  # Todo: what about this?
        progress = curr / total
        progress_bar = round(progress * steps)
        print('\r', '#' * progress_bar + '-' * (steps - progress_bar),
              '[{:.0%}] [{:.2f}s]'.format(progress, time), end='')
        sys.stdout.flush()  # Force flush to stdout

