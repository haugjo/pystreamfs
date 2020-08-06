from pystreamfs.evaluate_feature_selection import EvaluateFeatureSelection
from pystreamfs.metrics.nogueira_metric import NogueiraStabilityMetric
from pystreamfs.metrics.predictive_metric import PredictiveMetric
from pystreamfs.feature_selectors.fires import FIRESFeatureSelector

from skmultiflow.data import FileStream
from skmultiflow.neural_networks import PerceptronMask
from sklearn.metrics import accuracy_score

stream = FileStream('../datasets/spambase.csv', target_idx=0)
stream.prepare_for_use()

predictor = PerceptronMask()

fs = FIRESFeatureSelector(n_total_ftr=stream.n_features,
                          n_selected_ftr=10,
                          lr_mu=0.01,
                          lr_sigma=0.01,
                          epochs=1,
                          batch_size=25,
                          model='probit')

stability = NogueiraStabilityMetric(sliding_window=10)

accuracy = PredictiveMetric.sklearn_metric(metric=accuracy_score, name='Accuracy')

evaluator = EvaluateFeatureSelection(max_samples=100000,
                                     batch_size=100,
                                     pretrain_size=100,
                                     pred_metrics=[accuracy],
                                     fs_metrics=[stability],
                                     output_file_path=None,
                                     live_plot=True,
                                     plot_scale=1)

evaluator.evaluate(stream, fs, predictor, predictor_name='Perceptron')

test = None  # breakpoint
