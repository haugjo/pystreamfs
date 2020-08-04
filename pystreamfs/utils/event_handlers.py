import sys
import json
from tabulate import tabulate
from timeit import default_timer as timer
from datetime import datetime

from skmultiflow.data.base_stream import Stream

from pystreamfs.metrics.fs_metric import FSMetric
from pystreamfs.metrics.predictive_metric import PredictiveMetric
from pystreamfs.feature_selectors.base_feature_selector import BaseFeatureSelector


def start_evaluation_routine(evaluator):
    """ Start evaluation routine

    :param evaluator: (EvaluateFeatureSelection) Evaluator object

    """
    _check_configuration(evaluator)
    if evaluator.pretrain_size > 0:
        _pretrain_predictive_model(evaluator)


def finish_iteration_routine(evaluator, samples):
    """ Finish one iteration routine

    :param evaluator: (EvaluateFeatureSelection) Evaluator object
    :param samples: (int) Size of current data batch

    """
    evaluator.iteration += 1
    evaluator.global_sample_count += samples
    _update_progress_bar(evaluator)


def finish_evaluation_routine(evaluator):
    """ Finish evaluation routine

    :param evaluator: (EvaluateFeatureSelection) Evaluator object

    """
    evaluator.data_stream.restart()
    _print_to_console(evaluator)
    if evaluator.output_file_path is not None:
        _save_to_json(evaluator)


def _check_configuration(evaluator):
    """ Check configurations and variables

    Prior to the evaluation, we check that critical variables have the correct data type

    :param evaluator: (EvaluateFeatureSelection) Evaluator object

    """
    if not isinstance(evaluator.stream, Stream):
        raise BaseException('Data stream must be of type skmultiflow.data.base_stream.Stream')

    if not isinstance(evaluator.feature_selector, BaseFeatureSelector):
        raise BaseException('Feature selection model must be of type pystreamfs.feature_selectors.base_feature_selector.BaseFeatureSelector')

    if not hasattr(evaluator.predictor.model, 'partial_fit'):
        raise BaseException('Predictive model must have a function partial_fit()')

    if not hasattr(evaluator.predictor.model, 'predict'):
        raise BaseException('Predictive model must have a function predict()')

    for metric in evaluator.pred_metrics:
        if not isinstance(metric, PredictiveMetric):
            raise BaseException('Predictive metrics must be of type pystreamfs.metrics.predictive_metrics.predictive_metric.PredictiveMetric')

    for metric in evaluator.fs_metrics:
        if not isinstance(metric, FSMetric):
            raise BaseException('Feature selection metrics must be of type pystreamfs.metrics.fs_metrics.fs_metric.FSMetric')


def _pretrain_predictive_model(evaluator):
    """ Pre-train the predictive model before starting the evaluation

    :param evaluator: (EvaluateFeatureSelection) Evaluator object

    """
    print('Pre-train {} with {} observation(s).'.format(evaluator.predictor.name, evaluator.pretrain_size))

    X, y = evaluator.data_stream.next_sample(evaluator.pretrain_size)

    # Fit model and increase sample count
    evaluator.predictor.model.partial_fit(X=X, y=y, classes=evaluator.data_stream.target_values)
    evaluator.global_sample_count += evaluator.pretrain_size


def _update_progress_bar(evaluator):
    """ Update the progress bar

    :param evaluator: (EvaluateFeatureSelection) Evaluator object

    """
    j = evaluator.global_sample_count / evaluator.max_samples
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))
    sys.stdout.flush()


def _print_to_console(evaluator):
    """ Print a summary of the evaluation to the console

    :param evaluator: (EvaluateFeatureSelection) Evaluator object

    """
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


def _save_to_json(evaluator):
    """ Save the evaluator object to a json file

    Only save attributes that do not contain a function

    :param evaluator: (EvaluateFeatureSelection) Evaluator object

    """
    file_name = evaluator.output_file_path + 'pystreamfs_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.json'
    print('Save results to "{}"'.format(file_name))

    with open(file_name, 'w') as fp:
        json.dump(vars(evaluator), fp)
