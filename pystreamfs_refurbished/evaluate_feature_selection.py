import sys
from timeit import default_timer as timer
import numpy as np
from tabulate import tabulate
from abc import ABCMeta

from skmultiflow.data.base_stream import Stream
from skmultiflow.core.base import ClassifierMixin

from pystreamfs_refurbished.metrics.time_metric import TimeMetric
from pystreamfs_refurbished.metrics.fs_metrics.fs_metric import FSMetric
from pystreamfs_refurbished.metrics.predictive_metrics.predictive_metric import PredictiveMetric
from pystreamfs_refurbished.exceptions import InvalidModelError
from pystreamfs_refurbished.feature_selectors.base_feature_selector import BaseFeatureSelector


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
                 pred_metric=None,
                 fs_metric=None,
                 streaming_features=None,
                 check_concept_drift=False,
                 output_file=None,
                 show_plot=True,
                 show_live_plot=False,
                 restart_stream=True):

        # Class internal attributes
        self._iteration = 1  # logical time/iteration of data stream
        self._start_time = 0  # global start time
        self._global_sample_count = 0  # count all seen samples

        # General parameters
        # self.n_sliding = n_wait  # Todo: consider using sliding window for performance measurement
        self.max_samples = max_samples  # max samples to draw from stream
        self.pretrain_size = pretrain_size  # number of samples to pretrain model at t=0
        self.batch_size = batch_size  # size of data batch at time t
        self.max_time = max_time  # max time to run the experiment
        self.output_file = output_file  # path of the output file
        self.results = dict()  # storage for all results and metrics

        # Data Stream related parameters
        self.stream = None
        self.restart_stream = restart_stream  # restarting stream at end of simulation

        # Feature Selection related parameters
        self.feature_selector = None  # placeholder
        self.feature_selector_metric = fs_metric  # Todo: think about more than one metric
        self.check_concept_drift = check_concept_drift

        self.streaming_features = streaming_features  # time/feature-pairs for simulation of streaming features
        if streaming_features is None:
            self.streaming_features = dict()

        # Prediction related parameters
        self.predictor = None  # placeholder
        self.predictor_metric = pred_metric  # Todo: think about more than one metric

        # Visualization related parameters
        self.show_plot = show_plot
        self.show_live_plot = show_live_plot

    def evaluate(self, stream, fs_model, predictive_model, predictive_model_name=None):
        """ Evaluate a feature selection algorithm
        In future: compare multiple feature selection algorithms
        """
        self._start_time = timer()  # start experiment

        self.stream = stream
        self.feature_selector = fs_model
        self.predictor = _BasePredictiveModel(name=predictive_model_name, model=predictive_model)  # Wrap scikit-multiflow evaluator

        # Specify true max samples
        if self.max_samples > self.stream.n_samples:
            self.max_samples = self.stream.n_samples

        # Validate class parameters
        self.__check_configuration()

        # Pretrain predictive model at time t=0
        if self.pretrain_size > 0:
            self.__pretrain_predictive_model()

        # Simulate stream with feature selection using prequential evaluation
        self.__train_and_test()

    def __check_configuration(self):  # Todo: enhance
        if not isinstance(self.stream, Stream):
            raise InvalidModelError('Specified data stream is not of type Stream (scikit-multiflow data type)')
        if not isinstance(self.feature_selector, BaseFeatureSelector):
            raise InvalidModelError('Specified feature selection model is not of type BaseFeatureSelector '
                                    '(pystreamfs data type)')
        if not isinstance(self.predictor.model, ClassifierMixin):
            raise InvalidModelError('Specified predictive model is not of type ClassifierMixin '
                                    '(scikit-multiflow data type)')
        if not isinstance(self.predictor_metric, PredictiveMetric):
            raise InvalidModelError('Specified predictive metric is not of type BaseMetric '
                                    '(pystreamfs data type)')
        if not isinstance(self.feature_selector_metric, FSMetric):
            raise InvalidModelError('Specified feature selection metric is not of type FSMetric(BaseMetric) '
                                    '(pystreamfs data type)')

    def __pretrain_predictive_model(self):
        print('Pre-training on {} sample(s).'.format(self.pretrain_size))  # Todo: what if no pretraining???

        x, y = self.stream.next_sample(self.pretrain_size)

        # Prediction WITHOUT feature selection and computation of metrics
        self.predictor.model.partial_fit(X=x, y=y, classes=self.stream.target_values)

        # Increase global sample count
        self._global_sample_count += self.pretrain_size

    def __train_and_test(self):
        """ Prequential evaluation """
        print('Evaluating...')
        while ((self._global_sample_count < self.max_samples) & (timer() - self._start_time < self.max_time)
               & (self.stream.has_more_samples())):
            try:
                # Load batch
                if self._global_sample_count + self.batch_size <= self.max_samples:
                    samples = self.batch_size
                else:
                    samples = self.max_samples - self._global_sample_count  # all remaining samples
                x, y = self.stream.next_sample(samples)

                if x is not None and y is not None:
                    # Get active features
                    if self._iteration in self.streaming_features and self.feature_selector.supports_streaming_features:
                        x = self.__sparsify_x(x, self.streaming_features[self._iteration])
                        print('Detected streaming features at t={}'.format(self._iteration))

                    # Feature Selection
                    start = timer()
                    self.feature_selector.weight_features(x, y)
                    self.feature_selector.comp_time.compute(start, timer())
                    self.feature_selector.select_features()
                    self.feature_selector_metric.compute(self.feature_selector)

                    # Concept Drift Detection by Feature Selector
                    if self.check_concept_drift and self.feature_selector.supports_concept_drift_detection:
                        self.feature_selector.detect_concept_drift(x, y)

                    # Sparsify batch x -> retain selected features
                    x = self.__sparsify_x(x, self.feature_selector.selection[-1])

                    # Testing
                    start = timer()
                    prediction = self.predictor.model.predict(x)
                    self.predictor.predictions.append(prediction)
                    self.predictor_metric.compute(y, prediction)
                    self.predictor.testing_time.compute(start, timer())

                    # Training
                    start = timer()
                    self.predictor.model.partial_fit(x, y, self.stream.target_values)
                    self.predictor.training_time.compute(start, timer())

                    # Update global sample count and iteration/logical time
                    self._iteration += 1
                    self._global_sample_count += samples
                    self.__update_progress_bar()

            except BaseException as exc:  # Todo: check where the exception happens
                print(exc)
                break

        # Flush file buffer, in case it contains data Todo: print results to file
        # self._flush_file_buffer()

        self.__evaluation_summary()

        if self.restart_stream:
            self.stream.restart()

    def __sparsify_x(self, x, retained_features):
        """Set given features to zero
        This is done to specify active features in a feature stream and to specify the currently active features
        Todo: this is not the clean way! Find alternative implementation
        """
        sparse_matrix = np.zeros(x.shape)
        sparse_matrix[:, retained_features] = x[:, retained_features]
        return sparse_matrix

    def __update_progress_bar(self):
        j = self._global_sample_count / self.max_samples
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))
        sys.stdout.flush()

    def __evaluation_summary(self):
        # Feature selection related results
        fs_stats = dict()
        fs_stats['name'] = self.feature_selector.name
        fs_stats['n_selected_ftr'] = self.feature_selector.n_selected_ftr
        fs_stats['selected_ftr_weights'] = self.feature_selector.weights[self.feature_selector.selection]
        fs_stats['weight_development'] = self.feature_selector.weight_development
        fs_stats['time'] = self.feature_selector.comp_time
        fs_stats[self.feature_selector_metric.name + '_measures'] = self.feature_selector_metric.measures
        fs_stats[self.feature_selector_metric.name + '_mean'] = self.feature_selector_metric.mean
        fs_stats[self.feature_selector_metric.name + '_var'] = self.feature_selector_metric.var
        self.results['fs'] = fs_stats

        # Prediction related results
        pred_stats = dict()
        pred_stats['name'] = self.predictor.name
        pred_stats['predictions'] = self.predictor.predictions
        pred_stats['testing_time'] = self.predictor.testing_time
        pred_stats['training_time'] = self.predictor.training_time
        pred_stats[self.predictor_metric.name + '_measures'] = self.predictor_metric.measures
        pred_stats[self.predictor_metric.name + '_mean'] = self.predictor_metric.mean
        pred_stats[self.predictor_metric.name + '_var'] = self.predictor_metric.var


class _BasePredictiveModel(metaclass=ABCMeta):
    """Private class as wrapper for scikit-multiflow evaluators"""

    def __init__(self, name, model):
        self.name = name
        self.model = model  # placeholder for scikit multiflow evaluation model
        self.predictions = []
        self.testing_time = TimeMetric()
        self.training_time = TimeMetric()
