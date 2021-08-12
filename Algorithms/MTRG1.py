from Algorithm import Algorithm
import numpy as np
from EA_tools.population import initial_population_uni
from EA_tools.Solutions import Solution
from EA_tools.evaluations import dimension_replacement_evaluation
from operator import attrgetter
from Operators.DE import rand_1, binomial_crossover


def set_cv(cv, populations):
    best_cluster = min(populations, key=attrgetter("best_fitness"))
    np.put(cv, best_cluster.dimensions, best_cluster.best_soln)


class Cluster:
    def __init__(self, population, dimensions):
        self.population = population
        self.dimensions = dimensions
        self.best_soln = None
        self.best_fitness = None

    def evaluate(self, function, cv):
        for sol in self.population:
            dimension_replacement_evaluation(sol, cv, self.dimensions, function)

        self.set_best_soln()

    def set_best_soln(self):
        self.best_soln = min(self.population, key=attrgetter("fitness"))
        self.best_fitness = self.best_soln.fitness

    def elitist_selection(self, population):
        for i in range(len(population)):

            if population[i].fitness <= self.population[i].fitness:
                self.population[i] = population[i]


class SpSolutions(Solution):
    def __init__(self, genotype):
        Solution.__init__(self, genotype)


class MTRG(Algorithm):
    def __init__(self, params):
        Algorithm.__init__(self)
        self.params = params

    def optimize(self, function, seed):
        rng = np.random.RandomState(seed)
        dims = np.asarray(list(range(self.params["D"])))
        self.params["bounds"] = np.asarray(self.params["bounds"])

        # Step 1. Group Variables
        grouping = self.params["grouping"].group()
        num_groups = len(grouping)

        # Step 2. Define Component Vector
        cv = rng.uniform(self.params["bounds"][0],
                        self.params["bounds"][1],
                        self.params["D"])

        # Step 3. Initial Sub-Problem Populations
        populations = [Cluster(initial_population_uni(self.params["pop_size"],
                                                      len(dims[grouping[i]]),
                                                      [self.params["bounds"][0][grouping[i]],
                                                       self.params["bounds"][1][grouping[i]]],
                                                      rng),
                                                    grouping[i]) for i in range(len(grouping))]

        # Step 4. Evaluate Each Population
        for cl in populations:
            cl.evaluate(function, cv)

        # Step 5. Update the Component Vector
        set_cv(cv, populations)

        # Step 4. Main Loop
        fe = self.params["pop_size"] * num_groups
        while fe < self.params["fitness_evaluations"]:

            if self.params["verbose"]["output"] and fe % self.params["verbose"]["mod"]==0:
                print("Best found:%.10f after %d function evaluations"%(function(cv), fe))

            # Step 4.1 Evolve Each Cluster
            for ci in populations:

                # Step 4.1.1 Generate Population of Mutant Vectors
                mutants = rand_1(self.params["f"], ci.population, self.params["pop_size"], rng)

                if rng.rand() < self.params["rmp"]:
                    best_solns = [cj.best_soln for cj in populations if cj is not ci]
                    idxs = rng.randint(num_groups-1, size=self.params["pop_size"])

                    mutants = [best_solns[idxs[i]] for i in range(self.params["pop_size"])]

                # Step 4.1.2 Generate Offspring Population
                offsprings = [binomial_crossover(self.params["cr"], ci.population[i], mutants[i], rng)
                              for i in range(self.params["pop_size"])]

                # Step 4.1.3 Evaluate Offspring Population
                offspring_cluster = Cluster(offsprings, ci.dimensions)
                offspring_cluster.evaluate(function, cv)

                fe += self.params["pop_size"]

                # Step 4.1.4 Elitist Selection
                ci.elitist_selection(offspring_cluster.population)
                ci.set_best_soln()

                set_cv(cv, populations)
