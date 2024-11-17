import numpy as np

DNA_SIZE = 10
POP_SIZE = 200
CROSS_RATE  = 0.2
MUTATION_RATE = 0.01
N_GENERATION = 200
X_BOUND = [0, 2]

def main():
    ga = MGA(DNA_size=DNA_SIZE, DNA_bound=[0, 1], cross_rate=CROSS_RATE, mutation_rate=MUTATION_RATE, pop_size=POP_SIZE)
    for _ in range(N_GENERATION): 
        DNA_prod, pred = ga.evolve(5)

def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x

class MGA():
    def __init__(self, DNA_size, POP_size, CROSS_rate, MUTATION_rate, DNA_bound):
        self.DNA_size = DNA_size
        self.POP_size = POP_size
        self.CROSS_rate = CROSS_rate
        self.MUTATION_rate = MUTATION_rate
        self.DNA_bound = DNA_bound
        self.pop = np.random.randint(*DNA_bound, size=(1, DNA_size)).repeat(POP_size, axis=0)

    def DNAtransform(self, pop):
        return pop.dot(2**(np.arange(0, self.DNA_size)[::-1]))/float(2**self.DNA_size-1)*X_BOUND[1]
    
    def fitness(self, pred):
        return pred
    
    def crossover(self, loser_winner):
        # if np.random.rand()<self.CROSS_rate 
            # cross = np.random.randint(0, 2, size=(1, self.DNA_size)).astype(np.bool_)
        cross_idx = np.empty(1, self.DNA_size).astype(np.bool_)
        for i in range(self.DNA_size):
            cross_idx[i] = True if np.random.rand()<self.CROSS_rate else False
        loser_winner[0, cross_idx] = loser_winner[1, cross_idx]
        return loser_winner
    
    def mutate(self, loser_winner):
        mutation_idx = np.empty((self.DNA_size,)).astype(np.bool_)
        for i in range(self.DNA_size):
            mutation_idx[i] = True if np.random.rand()<self.MUTATION_rate else False
        loser_winner[0, mutation_idx] = ~loser_winner[0, mutation_idx].astype(np.bool_)
        return loser_winner
    
    def evolve(self, n):
        for _ in range(n):
            sub_pop_idx = np.random.choice(np.arange(0, self.pop_size), size=2, replace=False)
            sub_pop = self.pop[sub_pop_idx]
            product = F(self.DNAtransform(sub_pop))
            fitness = self.fitness(product)
            loser_winner_idx = np.argsort(fitness)
            loser_winner = sub_pop[loser_winner_idx]
            loser_winner = self.crossover(loser_winner)
            loser_winner = self.mutate(loser_winner)
            self.pop[sub_pop_idx] = loser_winner
        DNA_prod = self.DNAtransform(self.pop)
        pred = F(DNA_prod)
        return DNA_prod, pred
    

