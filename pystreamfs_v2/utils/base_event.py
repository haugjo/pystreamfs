class Event(list):
    """Base Event Class
    Todo: Use for visualization (event finishedIteration) and summary print (event finishedEvaluation)
    """
    def __call__(self, *args, **kwargs):
        for f in self:
            f(*args, **kwargs)

    def __repr__(self):
        return "Event(%s)" % list.__repr__(self)



