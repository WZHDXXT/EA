import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
import sys
budget = 50000
dimension = 10
DNA_BOUND = [-5, 5]

# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`
# Initialization
np.random.seed(42)

def initialization_individual_sigma(mu, dimension, lowerbound = -5.0, upperbound = 5.0):
    parent = []
    parent_sigma = []
    for i in range(mu):
        parent.append(np.random.uniform(low = lowerbound, high = upperbound, size = dimension))
        parent_sigma.append(
            0.05 +  0.02 * np.random.rand(dimension)
        )
    return np.array(parent), np.array(parent_sigma)



def initialization_one_sigma(mu, dimension, lowerbound = -5.0, upperbound = 5.0):
    parent = []
    parent_sigma = []
    for i in range(mu):
        parent.append(np.random.uniform(low = lowerbound, high = upperbound, size = dimension))
        parent_sigma.append(
            np.full(dimension, 0.05 +  0.02 * np.random.rand())
        )
    return np.array(parent), np.array(parent_sigma)

def one_sigma_mutation(parent, parent_sigma, tau):
    parent_sigma *= np.exp(np.random.normal(0, tau, size=parent_sigma.shape))
    parent += np.random.normal(0, parent_sigma)
    parent = np.clip(parent, -5.0, 5.0)

    return parent, parent_sigma


def individual_sigma_mutation(parent, parent_sigma, global_tau, local_tau):
    for kv, ks in zip(parent, parent_sigma):
        ks[:] *= (np.exp(np.random.normal(0, global_tau) + np.random.normal(0, local_tau)))
        kv += ks * np.random.randn(*kv.shape)
        kv[:] = np.clip(kv, *DNA_BOUND)
    return parent, parent_sigma


# Intermediate recombination
def intermediate_recombination(parent, parent_sigma):
    p1, p2 = np.random.choice(len(parent), 2, replace = False)
    offspring = (parent[p1] + parent[p2])/2
    sigma = (parent_sigma[p1] + parent_sigma[p2])/2 
    return offspring, sigma

def discrete_recombination(parent, parent_sigma):
    offspring = np.empty((1, len(parent[0])))
    offspring_sigma = np.empty((1, len(parent[0])))
    for kv, ks in zip(offspring, offspring_sigma):
        p1, p2 = np.random.choice(len(parent), size=2, replace=False)
        cp = np.random.randint(0, 2, len(parent[0]), dtype=np.bool_)
        kv[cp] = parent[p1, cp]
        kv[~cp] = parent[p2, ~cp]
        ks[cp] = parent_sigma[p1, cp]
        ks[~cp] = parent_sigma[p2, ~cp]
    return offspring, offspring_sigma

def global_discrete_recombination(parent, parent_sigma):
    offspring = np.empty((1, len(parent[0])))
    offspring_sigma = np.empty((1, len(parent[0])))
    for i in range(len(parent[0])):
        selected_parent = np.random.randint(0, len(parent)) 
        offspring[0, i] = parent[selected_parent, i] 
        offspring_sigma[0, i] = parent_sigma[selected_parent, i]
    return offspring, offspring_sigma

def global_intermediate_recombination(parent, parent_sigma):
    offspring = np.mean(parent, axis=0, keepdims=True) 
    offspring_sigma = np.mean(parent_sigma, axis=0, keepdims=True) 
    return offspring, offspring_sigma

def s4312090_s4406559_ES(problem):
    # hint: F18 and F19 are Boolean problems. Consider how to present bitstrings as real-valued vectors in ES
    # initial_pop = ... make sure you randomly create the first population
    optimum = problem.optimum.y
    
    f_opt = sys.float_info.max
    x_opt = None

    # Parameters setting 
    mu_ = 10
    lambda_ = 100
    learning_rate =  1.0 / np.sqrt(problem.meta_data.n_variables)
    
    local_learning_rate =  1.0 / np.sqrt(2* np.sqrt(problem.meta_data.n_variables))
    global_learning_rate = 1.0 / np.sqrt(2* problem.meta_data.n_variables)

    # Initialization and Evaluation
    parent, parent_sigma = initialization_individual_sigma(mu_, problem.meta_data.n_variables)
    
    problem_f = np.array(problem(parent))
    parent_f = np.abs(problem_f - optimum)
    '''if parent_f[i] < f_opt:
        f_opt = parent_f[i]
        # print('f_opt', f_opt)
        x_opt = parent[i].copy()'''
    
    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:
        offspring = np.empty((lambda_, parent[0].shape[0])) 
        offspring_sigma = np.empty_like(offspring) 
        offspring_f = np.empty(lambda_)

        # Recombination
        for i in range(lambda_):
            o, s = discrete_recombination(parent, parent_sigma)
            offspring[i] = o
            offspring_sigma[i] = s

        # Mutation
        # one_sigma_mutation(offspring, offspring_sigma, global_tau)
        offspring, offspring_sigma = individual_sigma_mutation(offspring, offspring_sigma, global_learning_rate, local_learning_rate)
        
        # Evaluation
        problem_f = np.array(problem(offspring))
        offspring_f = np.abs(problem_f - optimum)

        # Selection
        # (u,)
        new_parent_f = np.array(offspring_f)
        fitness_sort = np.argsort(new_parent_f)
        parent = offspring[fitness_sort][:mu_]
        parent_sigma = offspring_sigma[fitness_sort][:mu_]
        



def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.BBOB)
    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="(μ, λ) μ=10 λ=100",  # name of your algorithm
        algorithm_info="Practical assignment part2 of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    F23, _logger = create_problem(23)
    for run in range(20): 
        s4312090_s4406559_ES(F23)
        
        # print(np.abs(F23.optimum.y - F23.state.current_best.y))
        
        F23.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder


