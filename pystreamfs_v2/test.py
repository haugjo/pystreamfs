from pystreamfs_v2.evaluate_feature_selection import EvaluateFeatureSelection
from pystreamfs_v2.metrics.fs_metrics.stability_metric import NogueiraStabilityMetric
from pystreamfs_v2.metrics.predictive_metrics.predictive_metric import PredictiveMetric
from pystreamfs_v2.feature_selectors.fires import FIRESFeatureSelector
from pystreamfs_v2.feature_selectors.ofs import OFSFeatureSelector
from pystreamfs_v2.feature_selectors.efs import EFSFeatureSelector
from pystreamfs_v2.feature_selectors.fsds import FSDSFeatureSelector
from pystreamfs_v2.feature_selectors.cancelout import CancelOutFeatureSelector

from skmultiflow.trees import HoeffdingTree, HATT, HAT
from skmultiflow.data import FileStream
from skmultiflow.neural_networks import PerceptronMask
from skmultiflow.meta import OnlineBoosting
from skmultiflow.bayes import NaiveBayes
from sklearn.metrics import accuracy_score

stream = FileStream('../datasets/spambase.csv', target_idx=0)
stream.prepare_for_use()

predictor = PerceptronMask()

"""
fs = FIRESFeatureSelector(n_total_ftr=stream.n_features,
                          n_selected_ftr=10,
                          model='probit')

fs = OFSFeatureSelector(n_total_ftr=stream.n_features,
                        n_selected_ftr=10)

fs = EFSFeatureSelector(n_total_ftr=stream.n_features,
                        n_selected_ftr=10)
                        
fs = FSDSFeatureSelector(n_total_ftr=stream.n_features,
                         n_selected_ftr=10)
                         
fs = CancelOutFeatureSelector(n_total_ftr=stream.n_features,
                              n_selected_ftr=10)
"""
fs = FIRESFeatureSelector(n_total_ftr=stream.n_features,
                          n_selected_ftr=10,
                          lr_mu=0.01,
                          lr_sigma=0.01,
                          epochs=1,
                          batch_size=25,
                          model='probit')

stability = NogueiraStabilityMetric(sliding_window=10)

accuracy = PredictiveMetric.sklearn_metric(metric=accuracy_score, name='Accuracy')

evaluator = EvaluateFeatureSelection(max_samples=100000, batch_size=100, pretrain_size=100, max_time=float("inf"),
                                     pred_metric=accuracy, fs_metric=stability, output_file_path=None, show_plot=True)

evaluator.evaluate(stream, fs, predictor, predictive_model_name='Perceptron')

test =None # breakpoint
