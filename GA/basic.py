import numpy as np

DNA_SIZE = 10
POP_SIZE = 10
CROSS_RATE = 0.8
MUTATION_RATE = 0.003
N_GENERATIONS = 20
X_BOUND = [0, 5]

def main():
    # 随机生成值为 0和1 的POP_SIZE*DNA_SIZE的数组
    pop = np.random.randint(0, 2, (1, DNA_SIZE)).repeat(POP_SIZE, axis=0)

    for _ in range(N_GENERATIONS):
        F_values = F(translateDNA(pop))
        fitness = get_fitness(F_values)
        pop = select(pop, fitness)
        pop_copy = pop.copy()
        for parent in pop:
            child = crossover(parent, pop_copy)
            child = mutate(child)
            parent[:] = child
        print(F_values)
# 生成曲线
def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x

# 计算适应度
def get_fitness(pred):
    return pred + 1e-3 - np.min(pred)

# 二进制转十进制
def translateDNA(pop):
    return pop.dot(2**np.arange(DNA_SIZE)[::-1]) / (2**DNA_SIZE-1)*X_BOUND[1]

def select(pop, fitness):
    # 在POP_SIZE里选择适应度最高的，可以重复选择
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p = fitness/fitness.sum())
    return pop[idx]

def crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        # 随机选择一个母亲
        i_ = np.random.randint(0, POP_SIZE, size=1)
        # 随机生成配对位置
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool_)
        # 交叉配对
        parent[cross_points] = pop[i_, cross_points]
    return parent

def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand()<MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child

main()
