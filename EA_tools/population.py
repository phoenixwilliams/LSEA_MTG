from Solutions import Solution
import numpy as np


def initial_population_uni(size, dimension, bounds, rng):
    l = [Solution(rng.uniform(bounds[0], bounds[1], dimension)) for _ in range(size)]
    return l
