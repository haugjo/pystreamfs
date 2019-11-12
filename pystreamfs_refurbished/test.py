from pystreamfs_refurbished.evaluate_feature_selection import EvaluateFeatureSelection
from pystreamfs_refurbished.metrics.fs_metrics.stability_metric import NogueiraStabilityMetric
from pystreamfs_refurbished.metrics.predictive_metrics.predictive_metric import PredictiveMetric
from pystreamfs_refurbished.feature_selectors.fire import FIREFeatureSelector

from skmultiflow.trees import HoeffdingTree
from skmultiflow.data import FileStream
from sklearn.metrics import accuracy_score

stream = FileStream('../datasets/spambase.csv', target_idx=0)
stream.prepare_for_use()

ht = HoeffdingTree()

fs = FIREFeatureSelector(n_total_ftr=stream.n_features,
                         n_selected_ftr=10,
                         sigma_init=1,
                         epochs=10,
                         batch_size=20,
                         lr_mu=0.1,
                         lr_sigma=0.1,
                         lr_weights=0.1,
                         lr_lamb=0.1,
                         lamb_init=1,
                         model='probit')

stability = NogueiraStabilityMetric(sliding_window=20)

accuracy = PredictiveMetric.sklearn_metric(metric=accuracy_score, name='Accuracy')

eval = EvaluateFeatureSelection(max_samples=100000, batch_size=100, pretrain_size=200, max_time=float("inf"),
                                pred_metric=accuracy, fs_metric=stability)

eval.evaluate(stream, fs, ht, predictive_model_name='hoeffding tree')
