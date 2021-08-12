class Solution:
    def __init__(self, genotype):
        self.genotype = genotype
        self.fitness = None

    def __len__(self):
        return len(self.genotype)

    def __add__(self, soln):
        if isinstance(soln, float):
            return Solution(self.genotype + soln)

        elif isinstance(soln, Solution):
            return Solution(self.genotype + soln.genotype)

    def __sub__(self, soln):
        if isinstance(soln, float):
            return Solution(self.genotype - soln)

        elif isinstance(soln, Solution):
            return Solution(self.genotype - soln.genotype)

    def __mul__(self, soln):
        if isinstance(soln, float):
            return Solution(self.genotype * soln)

        elif isinstance(soln, Solution):
            return Solution(self.genotype * soln.genotype)

    def __getitem__(self, item):
        return self.genotype[item]

    def __str__(self):
        return str(self.genotype)

    def set_fitness(self, f):
        self.fitness = f
