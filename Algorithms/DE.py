from Algorithm import Algorithm
from operator import attrgetter
from Operators.DE import rand_1, binomial_crossover
from EA_tools.population import initial_population_uni
import numpy as np


class Cluster:
    def __init__(self, population):
        self.population = population
        self.best_soln = None
        self.best_fitness = None

    def evaluate(self, function):
        for sol in self.population:
            sol.set_fitness(function(sol.genotype))

        self.set_best_soln()

    def set_best_soln(self):
        self.best_soln = min(self.population, key=attrgetter("fitness"))
        self.best_fitness = self.best_soln.fitness

    def elitist_selection(self, population):
        for i in range(len(population)):

            if population[i].fitness <= self.population[i].fitness:
                self.population[i] = population[i]


class DE(Algorithm):
    def __init__(self, params):
        Algorithm.__init__(self)
        self.params = params

    def optimize(self, function, seed):
        rng = np.random.RandomState(seed)
        self.params["bounds"] = np.asarray(self.params["bounds"])

        # Initial Population and Evaluate
        cl = Cluster(initial_population_uni(self.params["pop_size"], self.params["D"], self.params["bounds"],
                                                    rng))
        cl.evaluate(function)
        cl.set_best_soln()


        # Main Loop
        for i in range(self.params["its"]):
            mutants = rand_1(self.params["f"], cl.population, self.params["pop_size"], rng)

            offcl = Cluster([binomial_crossover(self.params["cr"], cl.population[i], mutants[i], rng)
                          for i in range(self.params["pop_size"])])

            offcl.evaluate(function)
            cl.elitist_selection(offcl.population)
            cl.set_best_soln()

        return cl.best_soln.fitness, cl.best_soln.genotype








