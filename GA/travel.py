import numpy as np

CROSS_RATE = 0.1
MUTATION_RATE = 0.02
DNA_SIZE = 20
POP_SIZE = 200
N_GENERATION = 300

class GA():
    def __init__(self, DNA_size, pop_size, CROSS_rate, MUTATION_rate):
        self.DNA_size = DNA_size
        self.pop_size = pop_size
        self.cross_rate = CROSS_rate
        self.mutation_rate = MUTATION_rate
        # 随意变更顺序
        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in pop_size])
    
    def get_fitness(self, x, y):
        distance = np.empty_like(x.shape[0], dtype=np.float64)
        for i, x_, y_ in enumerate(zip(x, y)):
            distance[i] = np.sum(np.sqrt(np.square(np.diff(x_)) + np.square(np.diff(y_))))
        fitness = np.exp(self.DNA_size*2/distance) + 1e-3
        return fitness, distance
    
    def transformDNA(self, DNA, city_position):
        x = np.empty_like(DNA, dtype = np.float64)
        y = np.empty_like(DNA, dtype = np.float64)
        for i, e in enumerate(DNA):
            coordinate = city_position[e]
            x[i, :] = coordinate[:, 0]
            y[i, :] = coordinate[:, 1]
        return x, y
    
    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness/fitness.sum())
        return self.pop[idx]
    
    def crossover(self, parent, pop):
        if np.random.rand()<self.cross_rate:
            i = np.random.randint(0, self.pop_size, size=1)
            cross = np.random.randint(0, 2, size=self.DNA_size).astype(np.bool_)
            keep_city = parent[~cross]  # False的值      
            swap_city = pop[i, np.isin(pop[i].ravel(), keep_city, invert=True)]
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent
    
    def mutation(self, child):
        for point in range(self.DNA_size):
            if np.random.rand()<self.mutation_rate:
                k = np.random.randint(0, self.DNA_size, size=1)
                swapA, swapB = child[point], child[k]
                child[point], child[k] = swapB, swapA
        return child

    def evolve(self, fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:
            child = self.crossover(parent, pop_copy)
            child = self.mutation(child)
            parent[:] = child
        self.pop = pop

def main(): 
    city_position = np.random.rand(20,2)
    ga = GA(DNA_size = DNA_SIZE, pop_size = POP_SIZE, CROSS_rate = CROSS_RATE, MUTATION_rate = MUTATION_RATE)
    for i in range(N_GENERATION):
        x, y = ga.transformDNA(ga.pop, city_position)
        fitness, distance = ga.get_fitness(x, y)
        print("The",i, "is", fitness)
        ga.evolve(fitness)