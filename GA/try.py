import numpy as np
'''mutation_idx = np.array([True, False, True, True, False])
print(mutation_idx)
loser_winner = np.random.rand(2, 5).astype(np.float32)
print(loser_winner)
print(~loser_winner[0, mutation_idx].astype(np.bool_))
loser_winner[0, mutation_idx] = ~loser_winner[0, mutation_idx].astype(np.bool_)
print(loser_winner)'''

'''DNA_BOUND = np.array([0, 5, 9, 8, 1, 2])
np.random.seed(1)
pop = dict(DNA=5 * np.random.rand(6, 10),
           mut_strength = np.random.rand(5, 10))
np.random.rand(*pop['DNA'][0].shape)
print(pop['DNA'])
print('--')
print(DNA_BOUND.argsort())
print(pop['DNA'][DNA_BOUND.argsort()])'''

pop = np.random.randint(0, 2, (1, 10)).repeat(5, axis=0)
print(pop)
for parent in pop:
    print(parent[:])