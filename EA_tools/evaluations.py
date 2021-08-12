import numpy as np


def dimension_replacement_evaluation(soln, cv, dimensions, function):
    og = np.copy(cv)
    np.put(og, dimensions, soln)
    soln.set_fitness(function(og))
