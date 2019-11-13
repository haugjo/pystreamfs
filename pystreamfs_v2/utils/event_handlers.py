import sys
from tabulate import tabulate
from timeit import default_timer as timer
import json
import numpy as np
from datetime import datetime


def update_data_buffer(evaluator):
    evaluator.data_buffer.set_elements(
        ftr_weights=evaluator.feature_selector.weights,
        ftr_selection=evaluator.feature_selector.selection,
        concept_drifts=evaluator.feature_selector.concept_drifts,
        fs_time_measures=evaluator.feature_selector.comp_time.measures,
        fs_time_mean=evaluator.feature_selector.comp_time.mean,
        fs_time_var=evaluator.feature_selector.comp_time.var,
        fs_metric_measures=evaluator.feature_selector_metric.measures,
        fs_metric_mean=evaluator.feature_selector_metric.mean,
        fs_metric_var=evaluator.feature_selector_metric.var,
        test_time_measures=evaluator.predictor.testing_time.measures,
        test_time_mean=evaluator.predictor.testing_time.mean,
        test_time_var=evaluator.predictor.testing_time.var,
        train_time_measures=evaluator.predictor.training_time.measures,
        train_time_mean=evaluator.predictor.training_time.measures,
        train_time_var=evaluator.predictor.training_time.measures,
        predictor_metric_measures=evaluator.predictor_metric.measures,
        predictor_metric_mean=evaluator.predictor_metric.mean,
        predictor_metric_var=evaluator.predictor_metric.var
    )


def update_progress_bar(evaluator):
    j = evaluator.global_sample_count / evaluator.max_samples
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))
    sys.stdout.flush()


def update_live_plot(evaluator):  # Todo
    pass


def summarize_evaluation(evaluator):
    _print_to_console(evaluator)
    if evaluator.output_file_path is not None:
        _save_to_json(evaluator)
    _print_final_plot(evaluator)


def _print_final_plot(evaluator):  # Todo
    pass


def _save_to_json(evaluator):
    results = dict()  # Todo: substitute by file buffer

    # Feature selection related results
    fs_stats = dict()
    fs_stats['name'] = evaluator.feature_selector.name
    fs_stats['n_selected_ftr'] = evaluator.feature_selector.n_selected_ftr
    fs_stats['n_total_ftr'] = evaluator.feature_selector.n_total_ftr
    fs_stats['weights'] = evaluator.feature_selector.weights[-1]

    # Change to float for json dump
    fs_stats['selection'] = list(np.asarray(evaluator.feature_selector.selection[-1]).astype('float'))
    fs_stats['concept_drifts'] = list(np.asarray(evaluator.feature_selector.concept_drifts).astype('float'))

    fs_stats['time_measures'] = evaluator.feature_selector.comp_time.measures
    fs_stats['time_mean'] = evaluator.feature_selector.comp_time.mean
    fs_stats['time_var'] = evaluator.feature_selector.comp_time.var
    fs_stats[evaluator.feature_selector_metric.name + '_measures'] = evaluator.feature_selector_metric.measures
    fs_stats[evaluator.feature_selector_metric.name + '_mean'] = evaluator.feature_selector_metric.mean
    fs_stats[evaluator.feature_selector_metric.name + '_var'] = evaluator.feature_selector_metric.var
    results['fs'] = fs_stats

    # Prediction related results
    pred_stats = dict()
    pred_stats['name'] = evaluator.predictor.name
    pred_stats['testing_time_measures'] = evaluator.predictor.testing_time.measures
    pred_stats['testing_time_mean'] = evaluator.predictor.testing_time.mean
    pred_stats['testing_time_var'] = evaluator.predictor.testing_time.var
    pred_stats['training_time_measures'] = evaluator.predictor.training_time.measures
    pred_stats['training_time_mean'] = evaluator.predictor.training_time.mean
    pred_stats['training_time_var'] = evaluator.predictor.training_time.var
    pred_stats[evaluator.predictor_metric.name + '_measures'] = evaluator.predictor_metric.measures
    pred_stats[evaluator.predictor_metric.name + '_mean'] = evaluator.predictor_metric.mean
    pred_stats[evaluator.predictor_metric.name + '_var'] = evaluator.predictor_metric.var
    results['prediction'] = pred_stats

    file_name = evaluator.output_file_path + 'pystreamfs_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.json'

    with open(file_name, 'w') as fp:
        json.dump(results, fp)

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