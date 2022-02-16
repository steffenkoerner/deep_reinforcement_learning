import numpy as np
import random

class GaussianNoise:
    def __init__(self, size, std_start, std_end=None, steps=None):
        if std_end is None:
            std_end = std_start
            steps = 1
        self.inc = (std_end - std_start) / float(steps)
        self.current_std = std_start
        self.std_end = std_end
        self.std_start = std_start
        self.size = size

    def __call__(self):
        self.current_std = np.clip(self.current_std - self.inc, self.std_end, self.std_start)
        return np.random.normal(0,self.current_std, self.size)

class GaussianProcess():
    def __init__(self, size, std):
        self.size = size
        self.std = std

    def sample(self):
        return np.random.randn(*self.size) * self.std()

class LinearIncrements:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end