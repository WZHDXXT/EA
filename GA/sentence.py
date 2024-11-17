import numpy as np

TARGET_PHRASE = "You got it!"
POP_SIZE = 300
CROSS_RATE = 0.4
MUTATION_RATE = 0.01
N_GENERATIONS = 1000

DNA_SIZE = len(TARGET_PHRASE)
# 字母用数字代替
TARGET_ASCII = np.fromstring(TARGET_PHRASE, dtype=np.uint8)
ASCII_BOUND = [32, 126]

class GA():
    def __init__(self, DNA_size, DNA_bound, cross_rate, mutation_rate, pop_size):
        self.DNA_size = DNA_size
        DNA_bound[1] += 1
        self.DNA_bound = DNA_bound
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.pop_size = pop_size

        self.pop = np.random.randint(*DNA_bound, size=(pop_size, DNA_size)).astype(np.int8)

    def transformDNA(self, DNA):
        # s = pop.dot(2**np.arange(DNA_SIZE)[::-1])/(2**DNA_SIZE-1)*ASCII_BOUND[1]
        return [chr(d) for d in DNA]
    
    def get_fitness(self):
        return (self.pop == TARGET_ASCII).sum(axis=1) + 1e-3
    
    def select(self):
        fitness = self.get_fitness()
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness/fitness.sum())
        return self.pop[idx]
    
    def crossover(self, parent, pop_copy):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)
            cross_points = np.random.randint(0, 2, size=self.DNA_size).astype(np.bool_)
            parent[cross_points] = pop_copy[i_, cross_points]
        return parent
    
    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutation_rate:
                child[point] = np.random.randint(*self.DNA_bound)
        return child

    def evolve(self):
        pop = self.select()
        pop_copy = pop.copy()
        for parent in pop:
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop

if __name__ == '__main__':
    ga = GA(DNA_size=DNA_SIZE, DNA_bound=ASCII_BOUND, cross_rate=CROSS_RATE, mutation_rate=MUTATION_RATE, pop_size=POP_SIZE)

    for generation in range(N_GENERATIONS):
        fitness = ga.get_fitness()
        best_DNA = ga.pop[np.argmax(fitness)]
        best_phrase = ga.transformDNA(best_DNA)
        print('Gen', generation, ': ', best_phrase)
        if best_phrase == TARGET_PHRASE:
            break
        ga.evolve()