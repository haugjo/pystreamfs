from skmultiflow.data import AGRAWALGenerator
from skmultiflow.data import RandomRBFGeneratorDrift
from skmultiflow.data import SineGenerator


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
