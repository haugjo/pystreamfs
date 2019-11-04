'''
# The first example demonstrates how to evaluate one model
from skmultiflow.data import SEAGenerator
from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation import EvaluatePrequential

# Set the stream
stream = SEAGenerator(random_state=1)
stream.prepare_for_use()

# Set the model
ht = HoeffdingTree()

# Set the evaluator

evaluator = EvaluatePrequential(max_samples=10000,
                                max_time=1000,
                                show_plot=True,
                                metrics=['accuracy', 'kappa', 'f1', 'recall', 'precision'],
                                data_points_for_classification=True)

# Run evaluation
evaluator.evaluate(stream=stream, model=ht, model_names=['HT'])
'''


# The second example demonstrates how to compare two models
from skmultiflow.data import SEAGenerator
from skmultiflow.trees import HoeffdingTree
from skmultiflow.bayes import NaiveBayes
from skmultiflow.evaluation import EvaluatePrequential

# Set the stream
stream = SEAGenerator(random_state=1)
stream.prepare_for_use()

# Set the models
ht = HoeffdingTree()
nb = NaiveBayes()

evaluator = EvaluatePrequential(max_samples=10000,
                                max_time = 1000,
                                show_plot = True,
                                metrics = ['accuracy', 'kappa'])

# Run evaluation
evaluator.evaluate(stream=stream, model=[ht, nb], model_names=['HT', 'NB'])


'''
# The third example demonstrates how to evaluate one model
# and visualize the predictions using data points.
# Note: You can not in this case compare multiple models
from skmultiflow.data import SEAGenerator
from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation import EvaluatePrequential
# Set the stream
stream = SEAGenerator(random_state=1)
stream.prepare_for_use()
# Set the model
ht = HoeffdingTree()
# Set the evaluator
evaluator = EvaluatePrequential(max_samples=200,
                                n_wait = 1,
                                pretrain_size = 1,
                                max_time = 1000,
                                show_plot = True,
                                metrics = ['accuracy'],
                                data_points_for_classification = True)

# Run evaluation
evaluator.evaluate(stream=stream, model=ht, model_names=['HT'])
'''