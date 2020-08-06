import matplotlib
from matplotlib.animation import FuncAnimation
from pystreamfs.visualization.visualizer import Visualizer

matplotlib.use("TkAgg")


class LivePlot(Visualizer, FuncAnimation):
    def __init__(self, evaluator):
        Visualizer.__init__(self, evaluator)
        FuncAnimation.__init__(self, self.fig, self.update, frames=self.gen_function(evaluator), blit=False, interval=1, repeat=False)

    @staticmethod
    def gen_function(evaluator):
        """ Generator for live plot

        This function corresponds to EvaluateFeatureSelection._test_then_train() but yields a new frame (current evaluator) at every iteration.

        :param evaluator: (EvaluateFeatureSelection) Evaluator object

        """
        while evaluator.global_sample_count < evaluator.max_samples:
            try:
                evaluator.one_training_iteration()
                yield evaluator
            except BaseException as exc:
                print(exc)
                break