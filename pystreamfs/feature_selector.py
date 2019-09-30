from pystreamfs.fs_algorithms import cancelout, efs, fsds, iufes, mcnn, ofs, random_benchmark


class FeatureSelector:
    def __init__(self, name, prop):
        self.name = name

        fs_algorithms = {
            'cancelout': cancelout.run_cancelout,
            'efs': efs.run_efs,
            'fsds': fsds.run_fsds,
            'iufes': iufes.run_iufes,
            'mcnn': mcnn.run_mcnn,
            'ofs': ofs.run_ofs,
            'random': random_benchmark.run_random_benchmark
        }

        self.algorithm = fs_algorithms[name]
        self.prop = prop
