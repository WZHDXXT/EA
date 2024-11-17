import numpy as np

DNA_SIZE = 5
MUTATION_RATE = 0.003
ITERATION = 100
def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x

def get_fitness(pre):
    return pre.flatten()

def make_kids(parent):
    parent_copy = np.copy(parent)
    for i in range(DNA_SIZE):
        if np.random.rand()<MUTATION_RATE:
            parent_copy[i] = ~parent_copy[i]
    return parent_copy
    

parent = np.random.randint(0, 2, size=DNA_SIZE)
for i in range(ITERATION):
    kids = make_kids(parent)
    if (get_fitness(F(kids))>get_fitness(F(parent))):
        parent = kids
    else:
        parent = parent
