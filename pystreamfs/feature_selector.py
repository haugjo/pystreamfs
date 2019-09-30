from pystreamfs.fs_algorithms import cancelout, efs, fsds, iufes, mcnn, ofs, random_benchmark


class FeatureSelector:
    def __init__(self, name, param):
        self.name = name

        fs_algorithms = {
            'cancelout': cancelout,
            'efs': efs,
            'fsds': fsds,
            'iufes': iufes,
            'mcnn': mcnn,
            'ofs': ofs,
            'random': random_benchmark
        }

        self.algorithm = fs_algorithms['name']
        self.param = param
