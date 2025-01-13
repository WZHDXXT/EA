from typing import Tuple 
import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
import ioh
from ioh import get_problem, logger, ProblemClass
budget = 5000

np.random.seed(42)
GA_POP_SIZE = 10
GA_MUTATION_RATE = 0.01
GA_CROSSOVER_RATE = 0.5
tournament_k = 5
def crossover(p1, p2, crossover_rate):
    if np.random.uniform(0, 1)<crossover_rate:
        cross_point = np.random.randint(0, 2, len(p1)).astype(np.bool_)
        p1[cross_point] = p2[cross_point]
    return p1

'''def crossover_random(parent, pop, crossover_rate):
    if np.random.uniform(0, 1)<crossover_rate:
        i = np.random.randint(0, len(pop), 1)
        
        # randomly select one from pop
        cross_point = np.random.randint(0, 2, len(parent)).astype(np.bool_)
        parent[cross_point] = pop[i][cross_point]
    return parent'''
def two_point_crossover(p1, p2, crossover_rate):
    if np.random.uniform(0, 1) < crossover_rate:
        point1, point2 = sorted(np.random.choice(range(len(p1)), 2, replace=False))
        child = np.concatenate([p1[:point1], p2[point1:point2], p1[point2:]])
        return child
    return p1.copy()

def mutation(p, mutation_rate):
    for i in range(len(p)):
        if np.random.uniform(0, 1)<mutation_rate:
            p[i] = 1-p[i]
    return p

# ******** MATING SELECTION *********

# ******** roulette_wheel **********
def mating_selection_roulette_wheel(parent, parent_f):
    fitness = np.sum(parent_f)
    p = parent_f/(fitness+1e-5)
    sum = [0]
    for i in range(len(p)):
        s = 0
        for j in range(i+1):
            s += p[j] 
        sum.append(s)
    parents = []
    for _ in range(len(parent)):
        roll = np.random.uniform(0,1)
        for i in range(1, len(p)):
            if roll>sum[i] and roll<sum[i+1]:
                parents.append(parent[i])
                break
            else:
                parents.append(parent[0])
                break
    return np.array(parents)
    
# *********** tourament *************
def mating_selection_tourament(parent, parent_f, tournament_k=5):
    tournament_k = min(tournament_k, len(parent))
    parents = []
    parent_f = np.array(parent_f)
    for _ in range(len(parent)):
        parent_index =  np.random.choice(len(parent), size = tournament_k, replace=False)
        parent = parent[parent_index]
        parent_f = parent_f[parent_index]
        new_parent = parent[np.argmax(parent_f)]
        parents.append(new_parent)
    return np.array(parents)

# ********** sort selection *************
def rank_selection(parent, parent_f):
    rank = np.argsort(parent_f)
    probabilities = np.arange(1, len(parent_f) + 1) / sum(range(1, len(parent_f) + 1))
    # choose with probability of rank
    selected_indices = np.random.choice(rank, size=len(parent), p=probabilities)
    parents = parent[selected_indices]
    return np.array(parents)

# ********** stochastic universal sampling ************
def SUS(parent, parent_f):
    fitness = np.sum(parent_f)
    point_distance = fitness / len(parent)
    start_point = np.random.uniform(0, point_distance)
    points = [start_point + i * point_distance for i in range(len(parent))]
    selected = []
    for point in points:
        i = 0
        while True:
            if i >= len(parent_f):  
                break
            point -= parent_f[i]
            if point > 0: 
                i += 1
            else:
                selected.append(parent[i])
                break
    return np.array(selected)

# ********* random selection **************
def random_selection(parent, parent_f):
    indices = np.random.choice(len(parent), len(parent), replace=True)
    return parent[indices]




# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

def GA_1(problem: ioh.problem.PBO, pop_size, mutation_rate, crossover_rate) -> None:
    # initial_pop = ... make sure you randomly create the first population
    
    dim = problem.meta_data.n_variables
    initial_pop = np.random.randint(0, 2, size=(pop_size, dim))

    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    
    parent = initial_pop
    parent_f = problem(parent)

    # 保留当前最优解的变量
    best_solution = parent[np.argmax(parent_f)]
    best_fitness = np.max(parent_f)

    while problem.state.evaluations < budget:
        # *********** compare different mating selections ***********
        parent = mating_selection_roulette_wheel(parent, parent_f)

        new_parent = []
        for i in range(len(parent)):
            for j in range(i + 1, len(parent)):
                offspring = crossover(parent[i], parent[j], crossover_rate)
                mutation(offspring, mutation_rate)
                new_parent.append(offspring)

        new_parent = np.array(new_parent)
        new_parent_f = problem(new_parent)

        # *********** 精英保留策略 ***********
        # 1. 找到父代中最优个体及其适应度
        current_best_index = np.argmax(parent_f)
        current_best_solution = parent[current_best_index]
        current_best_fitness = parent_f[current_best_index]

        if current_best_fitness > best_fitness:
            best_solution = current_best_solution
            best_fitness = current_best_fitness

        # 2. 合并父代和子代
        combined_population = np.vstack([parent, new_parent])
        combined_fitness = np.hstack([parent_f, new_parent_f])

        # 3. 确保精英个体加入种群
        if best_solution not in combined_population:
            worst_index = np.argmin(combined_fitness)  # 找到最差个体
            combined_population[worst_index] = best_solution  # 替换为最优个体
            combined_fitness[worst_index] = best_fitness

        # 4. 根据适应度排序，选择前 GA_POP_SIZE 个个体作为新一代父代
        fitness_sort = np.argsort(combined_fitness)[::-1]  # 从高到低排序
        parent = combined_population[fitness_sort][:pop_size]
        parent_f = combined_fitness[fitness_sort][:pop_size]

def studentnumber1_studentnumber2_GA(problem: ioh.problem.PBO) -> None:
    # initial_pop = ... make sure you randomly create the first population
    
    dim = problem.meta_data.n_variables
    initial_pop = np.random.randint(0, 2, size=(GA_POP_SIZE, dim))

    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    
    parent = initial_pop
    parent_f = problem(parent)

    best_solution = parent[np.argmax(parent_f)]
    best_fitness = np.max(parent_f)

    while problem.state.evaluations < budget:
        # *********** compare different mating selections ***********
        parent = mating_selection_roulette_wheel(parent, parent_f)

        new_parent = []
        for i in range(len(parent)):
            for j in range(i + 1, len(parent)):
                offspring = crossover(parent[i], parent[j], GA_CROSSOVER_RATE)
                mutation(offspring, GA_MUTATION_RATE)
                new_parent.append(offspring)

        new_parent = np.array(new_parent)
        new_parent_f = problem(new_parent)

        # *********** 精英保留策略 ***********
        # 1. 找到父代中最优个体及其适应度
        current_best_index = np.argmax(parent_f)
        current_best_solution = parent[current_best_index]
        current_best_fitness = parent_f[current_best_index]

        if current_best_fitness > best_fitness:
            best_solution = current_best_solution
            best_fitness = current_best_fitness

        # 2. 合并父代和子代
        combined_population = np.vstack([parent, new_parent])
        combined_fitness = np.hstack([parent_f, new_parent_f])

        # 3. 确保精英个体加入种群
        if best_solution not in combined_population:
            worst_index = np.argmin(combined_fitness)  # 找到最差个体
            combined_population[worst_index] = best_solution  # 替换为最优个体
            combined_fitness[worst_index] = best_fitness

        # 4. 根据适应度排序，选择前 GA_POP_SIZE 个个体作为新一代父代
        fitness_sort = np.argsort(combined_fitness)[::-1]  # 从高到低排序
        parent = combined_population[fitness_sort][:GA_POP_SIZE]
        parent_f = combined_fitness[fitness_sort][:GA_POP_SIZE]



def create_problem(dimension: int, fid: int) -> Tuple[ioh.problem.PBO, ioh.logger.Analyzer]:
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="eitism",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    # create the LABS problem and the data logger
    F18, _logger = create_problem(dimension=50, fid=18)
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F18)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder
    
    # create the N-Queens problem and the data logger
    F23, _logger = create_problem(dimension=49, fid=23)
    for run in range(20): 
        studentnumber1_studentnumber2_GA(F23)
        F23.reset()
    _logger.close()