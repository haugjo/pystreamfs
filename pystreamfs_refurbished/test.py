from pystreamfs_refurbished.evaluate_feature_selection import EvaluateFeatureSelection
from pystreamfs_refurbished.metrics.stability_metric import NogueiraStabilityMetric
from pystreamfs_refurbished.metrics.base_metric import BaseMetric
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

stability = NogueiraStabilityMetric(20)

Accuracy = type('ScikitMetric', (BaseMetric,), {'compute': lambda self, true, predicted: self.measures.append([accuracy_score(true, predicted)])})
accuracy = Accuracy()

eval = EvaluateFeatureSelection(max_samples=100000, batch_size=100, pretrain_size=200, max_time=float("inf"),
                                predictive_metric=accuracy, fs_metric=stability)

eval.evaluate(stream, fs, ht, fs_model_name='fire', predictive_model_name='hoeffding tree')
