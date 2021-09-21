from Algorithms.DE import DE
import numpy as np


def ackley(x):
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)

    sum1 = 0
    sum2 = 0
    for i in range(d):
        sum1 += x[i]**2
        sum2 += np.cos(c * x[i])

    return -a * np.exp(-b * np.sqrt((1./d) * sum1)) - np.exp((1./d) * sum2) + a + np.exp(1.)

if __name__ == "__main__":
    dim = 10
    bounds = [-5, 5]
    bounds = [[bounds[0]] * dim, [bounds[1]] * dim]
    params = {
        'D': dim,
        "bounds": bounds,
        "pop_size": 100,
        "its": 100,
        "f": 0.6,
        "cr": 0.9
    }
    alg = DE(params)
    f, _ = alg.optimize(ackley, 0)

    print(f)
