from pystreamfs_refurbished.metrics import BaseMetric


class TimeMetric(BaseMetric):
    def __init__(self):
        super().__init__(self)

    def compute(self, start, end):
        self.measures.extend(end - start)
