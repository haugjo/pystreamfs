from skmultiflow.data import AGRAWALGenerator
from skmultiflow.data import RandomRBFGeneratorDrift
from skmultiflow.data import SineGenerator
from sklearn import preprocessing


class DataGenerator:
    def __init__(self, name):
        self.name = name

        switcher = {

            'agrawal': AGRAWALGenerator(),

            'rbf': RandomRBFGeneratorDrift(),

            'sine': SineGenerator()
        }

        self.multiflow_alg = switcher.get(name)
        self.multiflow_alg.prepare_for_use()

        # save number of generated features
        X, _ = self.multiflow_alg.next_sample(1)
        self.no_features = X.shape[1]

    def create_sample(self, n):
        X, Y = self.multiflow_alg.next_sample(n)

        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)

        return X, Y
