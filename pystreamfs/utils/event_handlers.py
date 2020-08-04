import sys
from tabulate import tabulate
from timeit import default_timer as timer
import json
from datetime import datetime

from skmultiflow.data.base_stream import Stream
from skmultiflow.core.base import ClassifierMixin

from pystreamfs.metrics.fs_metrics.fs_metric import FSMetric
from pystreamfs.metrics.predictive_metrics.predictive_metric import PredictiveMetric
from pystreamfs.utils.exceptions import InvalidModelError
from pystreamfs.feature_selectors.base_feature_selector import BaseFeatureSelector


def start_evaluation_routine(evaluator):
    """ Start evaluation routine

    :param evaluator: (EvaluateFeatureSelection) Evaluator object

    """
    _check_configuration(evaluator)
    _init_data_buffer(evaluator)
    if evaluator.pretrain_size > 0:
        _pretrain_predictive_model(evaluator)


def finish_iteration_routine(evaluator, samples):
    """ Finish one iteration routine

    :param evaluator: (EvaluateFeatureSelection) Evaluator object
    :param samples: (int) Size of current data batch

    """
    evaluator.iteration += 1
    evaluator.global_sample_count += samples
    _update_data_buffer(evaluator)
    _update_progress_bar(evaluator)


def finish_evaluation_routine(evaluator):
    """ Finish evaluation routine

    :param evaluator: (EvaluateFeatureSelection) Evaluator object

    """
    evaluator.data_stream.restart()
    _update_data_buffer(evaluator)
    _summarize_evaluation(evaluator)


def _check_configuration(evaluator):
    if not isinstance(evaluator.stream, Stream):
        raise InvalidModelError('Specified data stream is not of type Stream (scikit-multiflow data type)')
    if not isinstance(evaluator.feature_selector, BaseFeatureSelector):
        raise InvalidModelError('Specified feature selection model is not of type BaseFeatureSelector '
                                '(pystreamfs data type)')
    if not isinstance(evaluator.predictor.model, ClassifierMixin):
        raise InvalidModelError('Specified predictive model is not of type ClassifierMixin '
                                '(scikit-multiflow data type)')
    if not isinstance(evaluator.predictor_metric, PredictiveMetric):
        raise InvalidModelError('Specified predictive metric is not of type BaseMetric '
                                '(pystreamfs data type)')
    if not isinstance(evaluator.feature_selector_metric, FSMetric):
        raise InvalidModelError('Specified feature selection metric is not of type FSMetric(BaseMetric) '
                                '(pystreamfs data type)')


def _init_data_buffer(evaluator):
    evaluator.data_buffer.set_elements(
        fs_name=evaluator.feature_selector.name,
        fs_metric_name=evaluator.feature_selector_metric.name,
        n_selected_ftr=evaluator.feature_selector.n_selected_ftr,
        n_total_ftr=evaluator.feature_selector.n_total_ftr,
        predictor_name=evaluator.predictor.name,
        predictor_metric_name=evaluator.predictor_metric.name,
        batch_size=evaluator.batch_size,
        max_samples=evaluator.max_samples,
        pretrain_size=evaluator.pretrain_size,
        iteration=evaluator.iteration
    )


def _pretrain_predictive_model(evaluator):
    print('Pre-train {} with {} observation(s).'.format(evaluator.predictor.name, evaluator.pretrain_size))

    X, y = evaluator.data_stream.next_sample(evaluator.pretrain_size)

    # Fit model and increase sample count
    evaluator.predictor.model.partial_fit(X=X, y=y, classes=evaluator.data_stream.target_values)
    evaluator.global_sample_count += evaluator.pretrain_size


def _update_data_buffer(evaluator):
    evaluator.data_buffer.set_elements(  # Todo: is it possible just to add the last element instead of saving the whole arrays again?
        ftr_weights=evaluator.feature_selector.weights.copy(),
        ftr_selection=evaluator.feature_selector.selection.copy(),
        concept_drifts=evaluator.feature_selector.concept_drifts.copy(),
        fs_time_measures=evaluator.feature_selector.comp_time.measures.copy(),
        fs_time_mean=evaluator.feature_selector.comp_time.mean.copy(),
        fs_time_var=evaluator.feature_selector.comp_time.var.copy(),
        fs_metric_measures=evaluator.feature_selector_metric.measures.copy(),
        fs_metric_mean=evaluator.feature_selector_metric.mean.copy(),
        fs_metric_var=evaluator.feature_selector_metric.var.copy(),
        test_time_measures=evaluator.predictor.testing_time.measures.copy(),
        test_time_mean=evaluator.predictor.testing_time.mean.copy(),
        test_time_var=evaluator.predictor.testing_time.var.copy(),
        train_time_measures=evaluator.predictor.training_time.measures.copy(),
        train_time_mean=evaluator.predictor.training_time.mean.copy(),
        train_time_var=evaluator.predictor.training_time.var.copy(),
        predictor_metric_measures=evaluator.predictor_metric.measures.copy(),
        predictor_metric_mean=evaluator.predictor_metric.mean.copy(),
        predictor_metric_var=evaluator.predictor_metric.var.copy(),
        predictions=evaluator.predictor.predictions.copy()
    )


def _update_progress_bar(evaluator):
    j = evaluator.global_sample_count / evaluator.max_samples
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))
    sys.stdout.flush()


def _summarize_evaluation(evaluator):
    _print_to_console(evaluator)
    if evaluator.output_file_path is not None:
        _save_to_json(evaluator)


def _save_to_json(evaluator):
    file_name = evaluator.output_file_path + 'pystreamfs_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.json'

    with open(file_name, 'w') as fp:
        json.dump(vars(evaluator.data_buffer), fp)

    print('Saved results to "{}"'.format(file_name))


def _print_to_console(evaluator):
    print('\n################################## SUMMARY ##################################')
    print('Evaluation finished after {}s'.format(timer() - evaluator.start_time))
    print('Processed {} instances in batches of {}'.format(evaluator.global_sample_count, evaluator.batch_size))
    print('----------------------')
    print('Feature Selection ({}/{} features):'.format(evaluator.feature_selector.n_selected_ftr, evaluator.feature_selector.n_total_ftr))
    print(tabulate({
        'Model': [evaluator.feature_selector.name],
        'Avg. Time': [evaluator.feature_selector.comp_time.mean],
        'Avg. {}'.format(evaluator.feature_selector_metric.name): [evaluator.feature_selector_metric.mean]
    }, headers="keys", tablefmt='github'))
    print('----------------------')
    print('Prediction:')
    print(tabulate({
        'Model': [evaluator.predictor.name],
        'Avg. Test Time': [evaluator.predictor.testing_time.mean],
        'Avg. Train Time': [evaluator.predictor.training_time.mean],
        'Avg. {}'.format(evaluator.predictor_metric.name): [evaluator.predictor_metric.mean]
    }, headers="keys", tablefmt='github'))
    print('#############################################################################')