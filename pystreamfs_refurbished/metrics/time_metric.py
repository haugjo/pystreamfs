from pystreamfs_refurbished.metrics.base_metric import BaseMetric


class TimeMetric(BaseMetric):
    def __init__(self):
        super().__init__(name='Time (ms)')

    def compute(self, start, end):
        self.measures.extend([end - start])
        super().compute()  # update sufficient statistics
