from pystreamfs_refurbished.metrics.base_metric import BaseMetric


class TimeMetric(BaseMetric):
    def __init__(self):
        super().__init__()

    def compute(self, start, end):
        self.measures.extend([end - start])
