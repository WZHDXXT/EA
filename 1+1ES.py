import numpy as np

MUT_STRENGTH = 0.1
DNA_SIZE = 3

def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x

def get_fitness(pre):
    return pre.flatten()

def make_kids(parent):
    k = parent + MUT_STRENGTH * np.random.randn(DNA_SIZE)
    return k

def kill_kids(parent, kid):
    global MUT_STRENGTH
    p_target = 1/5
    p1 = get_fitness(F(parent))
    p2 = get_fitness(F(kid))
    if p1 < p2:
        parent = kid
        ps = 1
    else:
        ps = 0
    MUT_STRENGTH *= np.exp((1/DNA_SIZE) * (ps-p_target)/(1-p_target))
    return parent

parent = 5 * np.random.rand(1, DNA_SIZE)
kids = make_kids(parent)
parent = kill_kids(parent, kids)