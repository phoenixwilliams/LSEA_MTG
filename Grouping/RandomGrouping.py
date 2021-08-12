import numpy as np
from Grouping import Grouping


class RandomGrouping(Grouping):
    def __init__(self, d, s):
        """

        :param d: problem dimension
        :param s:
        """
        self.d = d
        self.s = s

    def group(self):
        dims = np.arange(self.d)
        return np.random.choice(dims, (self.s, self.d // self.s), replace=False)
