from EA_tools.Solutions import Solution
import numpy as np


def rand_1(f, solutions, popsize, rng):

    mutants = [None] * popsize
    for mi in range(popsize):
        idxs = [idx for idx in range(popsize) if idx != mi]
        x1, x2, x3 = rng.choice(idxs, 3, replace=False)
        mutants[mi] = solutions[x1] + (solutions[x2] - solutions[x3]) * f

    return mutants


def binomial_crossover(cr, sol, mutant, rng):

    cross_points = rng.rand(len(sol)) < cr

    if not np.any(cross_points):
        cross_points[rng.randint(0, len(sol))] = True

    return Solution(np.where(cross_points, mutant, sol))

