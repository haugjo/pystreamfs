import sys
from timeit import default_timer as timer
import numpy as np

from skmultiflow.data.base_stream import Stream
from skmultiflow.core.base import ClassifierMixin

from pystreamfs_refurbished.metrics.time_metric import TimeMetric
from pystreamfs_refurbished.metrics.fs_metrics.fs_metric import FSMetric
from pystreamfs_refurbished.metrics.predictive_metrics.predictive_metric import PredictiveMetric
from pystreamfs_refurbished.exceptions import InvalidModelError
from pystreamfs_refurbished.feature_selectors.base_feature_selection import BaseFeatureSelector


class EvaluateFeatureSelection:
    """ Evaluation of online Feature Selection algorithms
    using the prequential evaluation method or interleaved test-then-train method.

    Reduced version of PrequentialEvaluation from scikit-multiflow, optimized for online feature selection
    """

    def __init__(self,
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
        self.predictions = []
        self.predictive_metric = predictive_metric  # Todo: think about more than one metric
        self.testing_time = TimeMetric()
        self.training_time = TimeMetric()

        # Visualization related parameters
        self.show_plot = show_plot
        self.show_live_plot = show_live_plot

        # Feature Selection related parameters
        self.fs_model = None
        self.fs_model_name = None
        self.fs_metric = fs_metric  # Todo: think about more than one metric
        self.fs_time = TimeMetric()

        self.streaming_features = streaming_features  # time/feature-pairs for simulation of streaming features
        if streaming_features is None:
            self.streaming_features = dict()

        self.active_features = None  # currently active features

        # Further general parameters
        self.time = 1  # logical time/iteration of data stream
        self._start_time = 0  # global start time
        self.global_sample_count = 0  # count all seen samples
        self.results = None  # storage for all results and metrics

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

        # Specify true max samples
        if self.max_samples > self.stream.n_samples:
            self.max_samples = self.stream.n_samples

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
        if not isinstance(self.predictive_model, ClassifierMixin):
            raise InvalidModelError('Specified predictive model is not of type ClassifierMixin '
                                    '(scikit-multiflow data type)')
        if not isinstance(self.predictive_metric, PredictiveMetric):
            raise InvalidModelError('Specified predictive metric is not of type BaseMetric '
                                    '(pystreamfs data type)')
        if not isinstance(self.fs_metric, FSMetric):
            raise InvalidModelError('Specified feature selection metric is not of type FSMetric(BaseMetric) '
                                    '(pystreamfs data type)')

    def __pretrain_predictive_model(self):
        print('Pre-training on {} sample(s).'.format(self.pretrain_size))  # Todo: what if no pretraining???

        x, y = self.stream.next_sample(self.pretrain_size)

        # Prediction WITHOUT feature selection and computation of metrics
        self.predictive_model.partial_fit(X=x, y=y, classes=self.stream.target_values)

        # Increase global sample count
        self.global_sample_count += self.pretrain_size

    def __train_and_test(self):
        """ Prequential evaluation """
        print('Evaluating...')
        while ((self.global_sample_count < self.max_samples) & (timer() - self._start_time < self.max_time)
               & (self.stream.has_more_samples())):
            try:
                # Load batch
                if self.global_sample_count + self.batch_size <= self.max_samples:
                    samples = self.batch_size
                else:
                    samples = self.max_samples - self.global_sample_count
                x, y = self.stream.next_sample(samples)

                if x is not None and y is not None:
                    # Streaming Features
                    if self.time in self.streaming_features:
                        self.active_features = self.streaming_features[self.time]
                    elif self.active_features is None:
                        self.active_features = np.arange(self.stream.n_features)  # all features are active

                    # Feature Selection
                    start = timer()
                    self.fs_model.weight_features(x, y, self.active_features)
                    self.fs_time.compute(start, timer())
                    self.fs_model.select_features()

                    self.fs_metric.compute(self.fs_model)

                    # Sparsify batch
                    sparse_matrix = np.zeros(x.shape)
                    sparse_matrix[:, self.fs_model.selection] = x[:, self.fs_model.selection]
                    x = sparse_matrix

                    # Testing
                    start = timer()
                    prediction = self.predictive_model.predict(x)
                    self.predictions.append(prediction)
                    self.predictive_metric.compute(y, prediction)
                    self.testing_time.compute(start, timer())

                    # Training
                    start = timer()
                    self.predictive_model.partial_fit(x, y, self.stream.target_values)
                    self.training_time.compute(start, timer())

                    # Update global sample count and time
                    self.time += 1
                    self.global_sample_count += samples
                    self.__update_progress_bar()

            except BaseException as exc:  # Todo: check where the exception happens
                print(exc)
                break

        # Flush file buffer, in case it contains data Todo: print results to file
        # self._flush_file_buffer()

        self.__evaluation_summary()

        if self.restart_stream:
            self.stream.restart()

    def __update_progress_bar(self):
        j = self.global_sample_count / self.max_samples
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))
        sys.stdout.flush()

    def __evaluation_summary(self):
        print('Finished Evaluation on {} samples.'.format(self.global_sample_count))
        print('Feature Selection results for {}'.format(self.fs_model_name))
        print('Final Feature Set (size {}): {}'.format(self.fs_model.n_selected_ftr, self.fs_model.selection[-1:]))
        print('Final feature weights: {}'.format(self.fs_model.weights))
        print('Avg. FS-metric ({}): {}'.format(type(self.fs_metric).__name__, self.fs_metric.get_mean()))
        print('Avg. Time for Feature Selection: {}'.format(self.fs_time.get_mean()))
        print('---------------------------------------')
        print('Prediction results for {}'.format(self.predictive_model_name))
        print('Avg. predictive metric ({}): {}'.format(type(self.predictive_model), self.predictive_metric.get_mean()))
        print('Avg. Testing Time: {}'.format(self.testing_time.get_mean()))
        print('Avg. Training Time: {}').format(self.training_time.get_mean())
