from Grouping.RandomGrouping import RandomGrouping
from Algorithms.MTRG1 import MTRG


def sphere(x):
    return sum(xi**2 for xi in x)


if __name__ == "__main__":
    dimension = 100
    bounds = [[-5] * dimension, [5]*dimension]

    params = {
        "grouping": RandomGrouping(dimension, 10),
        "bounds": bounds,
        "D": dimension,
        "pop_size": 20,
        "fitness_evaluations": 1e+5,
        "f": 0.6,
        "cr": 0.9,
        "rmp": 0.3,
        "verbose": {"output": True, "mod": 3e+4}
    }
    algorithm = MTRG(params)
    algorithm.optimize(sphere, 0)



