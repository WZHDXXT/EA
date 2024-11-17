import numpy as np

DNA_SIZE = 1
DNA_BOUND = [0, 5]
POP_SIZE = 5
N_GENERATION = 3
N_KID = 5

def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x

def get_fitness(pred):
    return pred.flatten()

def make_kid(pop, n_kid):
    kids = {'DNA':np.empty((n_kid, DNA_SIZE))}
    kids['mut_strength'] = np.empty_like(kids['DNA'])
    for kv, ks in zip(kids['DNA'], kids['mut_strength']):
        p1, p2 = np.random.choice(np.arange(POP_SIZE), size=2, replace=False)
        cp = np.random.randint(0, 2, DNA_SIZE, dtype=np.bool_)
        kv[cp] = pop['DNA'][p1, cp]
        kv[~cp] = pop['DNA'][p2, ~cp]
        ks[cp] = pop['mut_strength'][p1, cp]
        ks[~cp] = pop['mut_strength'][p2, ~cp]

        # mutate (change DNA based on normal distribution)
        ks[:] = np.maximum(ks + (np.random.rand(*ks.shape)-0.5), 0.)    # must > 0
        kv += ks * np.random.randn(*kv.shape)
        kv[:] = np.clip(kv, *DNA_BOUND)    # clip the mutated value
    return kids

def kill_kid(pop, kids):
    for key in ['DNA', 'mut_strength']:
        pop[key] = np.vstack((pop[key], kids[key]))
    fitness = get_fitness(F(pop['DNA']))
    idx = fitness.argsort()
    for key in ['DNA', 'mut_strength']:
        pop[key] = pop[key][idx][:-POP_SIZE]
    return pop

pop = dict(DNA=5 * np.random.rand(1, DNA_SIZE).repeat(POP_SIZE, axis=0),
           mut_strength = np.random.rand(POP_SIZE, DNA_SIZE))
for _ in range(N_GENERATION):
    kids = make_kid(pop, N_KID)
    pop = kill_kid(pop, kids)