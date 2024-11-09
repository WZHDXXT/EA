from typing import List
import ConfigSpace
import numpy as np
from scipy.stats import norm
import sklearn.gaussian_process
from sklearn.gaussian_process.kernels import RBF, WhiteKernel,ConstantKernel as C

# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
from GA import studentnumber1_studentnumber2_GA, GA_1, create_problem
from skopt import gp_minimize
budget = 1000000
# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

# Hyperparameters to tune, e.g.
hyperparameter_space = {
    "population_size": [50, 100, 200],
    "mutation_rate": [0.01, 0.05, 0.1],
    "crossover_rate": [0.5, 0.7, 0.9]
}
d = 3  
num = 5 
origin = [55, 0.06, 0.6] 
radius = [45, 0.04, 0.3]

# Hyperparameter tuning function
def tune_hyperparameters() -> List:
    # You should decide/engineer the `score` youself, which is the tuning objective
    best_score = float('inf')
    best_params = None
    # create the LABS problem and the data logger
    F18, _logger_18 = create_problem(dimension=50, fid=18)
    # create the N-Queens problem and the data logger
    F23, _logger_23 = create_problem(dimension=49, fid=23)
    
    
  # ********** optimization ***********
    sample_budget = 10
    sampler_1 = RandomSampler(GA_evaluation, d, sample_budget, F18, F23)
    sampler_1.run(budget=1)
    
    sampler_2 = TuRBO(GA_evaluation, d, sample_budget, F18, F23)
    sampler_2.run(budget=5)

    sampler_3 = TuRBO(GA_evaluation, d, sample_budget, F18, F23)
    sampler_3.run(budget=100)
    samplers = [sampler_1, sampler_2, sampler_3]
    best_overall_x = None
    best_overall_y = -float('inf')


    # ******** print samplers' best_y to compare **********
    for sampler in samplers:
        if sampler.best_y[-1] > best_overall_y:
            best_overall_y = sampler.best_y[-1]
            best_overall_x = sampler.best_x[-1]

    return best_overall_x
    
    '''print(sampler_2.best_y)
    print(sampler_2.best_x[-1])
    return sampler_2.best_x[-1]'''

def GA_evaluation(problem1, problem2, sample):
    pop_size, mutation_rate, crossover_rate = sample
    pop_size = round(pop_size)
    mutation_rate = round(mutation_rate, 2)
    crossover_rate = round(crossover_rate, 1)
    GA_1(problem1, pop_size, mutation_rate, crossover_rate)
    # print(problem1.state.current_best)
    F_18 = problem1.state.current_best.y
    problem1.reset()
    GA_1(problem2, pop_size, mutation_rate, crossover_rate)
    F_23 = problem2.state.current_best.y
    # print(problem2.state.current_best)
    problem2.reset()
    
    # ********** set different weights after recording the initial results *******
    score = (F_18 + F_23)/2
    return score


def random_sample(d, num, origin=origin, radius=radius):
    origin = np.array(origin)
    radius = np.array(radius)
    return np.random.rand(num, d) * 2 * radius + origin - radius

class RandomSampler:
    def __init__(self, problem, d, doe_sample_budget, *problem_args, is_reset=False):
        self.problem = problem
        self.d = d
        self.obs_x = []
        self.obs_y = []
        self.best_x = []
        self.problem_args = problem_args

        if not is_reset:
            self.best_y = []
        doe = random_sample(d, doe_sample_budget)
        for sample in doe:
            self.evaluate(sample)

    def evaluate(self, sample):
        score = self.problem(*self.problem_args, sample)
        self.obs_x.append(sample)
        self.obs_y.append(score)
        if self.best_y:
            if score > self.best_y[-1]:
                self.best_x.append(sample)
                self.best_y.append(score)
        else:
            self.best_x.append(sample)
            self.best_y.append(score)

    def run(self, budget):
        for _ in range(budget):
            self.evaluate(random_sample(self.d, 1)[0])

class GlobalBO(RandomSampler):
    def __init__(self, problem, d, doe_sample_budget, *problem_args, **kwargs):
        super().__init__(problem, d, doe_sample_budget, *problem_args, **kwargs)

        kernel = None
        self.gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel)
        self.proxy_sample_size = 10

    def run(self, budget):
        for _ in range(budget):
            # fit GP model
            self.gp.fit(self.obs_x, self.obs_y)

            # Thompson sample
            proxy_sample_x = random_sample(self.d, self.proxy_sample_size)
            proxy_sample_y = self.gp.sample_y(proxy_sample_x).reshape(-1)
            best_proxy_x = proxy_sample_x[min(range(self.proxy_sample_size), key=lambda i: proxy_sample_y[i])]

            self.evaluate(best_proxy_x)

class GlobalBO_EI(RandomSampler):
    def __init__(self, problem, d, doe_sample_budget, *problem_args, **kwargs):
        super().__init__(problem, d, doe_sample_budget, *problem_args, **kwargs)
        
        kernel = sklearn.gaussian_process.kernels.Matern()
        self.gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel)
        self.proxy_sample_size = 10

    def expected_improvement(self, x, gp, y_best):
        mean, std = gp.predict(x, return_std=True)
        with np.errstate(divide='warn'):
            improvement = y_best - mean
            z = improvement / std
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei[std == 0.0] = 0.0 
        return ei

    def run(self, budget):
        for _ in range(budget):
            self.gp.fit(self.obs_x, self.obs_y)
            
            proxy_sample_x = random_sample(self.d, self.proxy_sample_size)
            y_best = min(self.obs_y)
            ei_values = self.expected_improvement(proxy_sample_x, self.gp, y_best)
            best_proxy_x = proxy_sample_x[np.argmax(ei_values)] 
            self.evaluate(best_proxy_x)


class TuRBO(GlobalBO):
    def __init__(self, problem, d, doe_sample_budget, *problem_args, **kwargs):
        super().__init__(problem, d, doe_sample_budget, *problem_args, **kwargs)

        self.doe_sample_budget = doe_sample_budget

        self.min_tr_rad = 0.01
        self.max_tr_rad = 1
        self.tr_succ_thresh = 5
        self.tr_fail_thresh = 5

        self.tr_rad = 0.25
        self.tr_succ_count = 0
        self.tr_fail_count = 0

    def run(self, budget):
        while budget > 0:
            best_x = self.obs_x[min(range(len(self.obs_y)), key=lambda i: self.obs_y[i])]
            tr_obs = [(obs_x, obs_y) for obs_x, obs_y in zip(self.obs_x, self.obs_y) if np.max(np.abs(obs_x - best_x)) <= self.tr_rad]
            self.gp.fit(*zip(*tr_obs))

            proxy_sample_x = random_sample(self.d, self.proxy_sample_size, origin=best_x, radius=self.tr_rad)

            # Thompson sampling
            proxy_sample_y = self.gp.sample_y(proxy_sample_x).reshape(-1)
            best_proxy_x = proxy_sample_x[min(range(self.proxy_sample_size), key=lambda i: proxy_sample_y[i])]

            self.evaluate(best_proxy_x)
            budget -= 1
            improvement = self.best_y[-1] > self.best_y[-2]
            
            if improvement:
                self.tr_succ_count += 1
            else:
                self.tr_fail_count += 1

            if self.tr_succ_count >= self.tr_succ_thresh:
                self.tr_rad = max(2 * self.tr_rad, self.max_tr_rad)
            elif self.tr_fail_count >= self.tr_fail_thresh:
                self.tr_rad /= 2
                if self.tr_rad < self.min_tr_rad:
                    doe_budget = min(self.doe_sample_budget, budget)
                    self.__init__(self.problem, self.d, doe_budget, is_reset=True)
                    budget -= doe_budget

            if self.tr_succ_count >= self.tr_succ_thresh or self.tr_fail_count >= self.tr_fail_thresh:
                self.tr_succ_count, self.tr_fail_count = 0, 0


class TuRBO_EI(GlobalBO):
    def __init__(self, problem, d, doe_sample_budget, *problem_args, **kwargs):
        super().__init__(problem, d, doe_sample_budget, *problem_args, **kwargs)

        self.doe_sample_budget = doe_sample_budget

        self.min_tr_rad = 0.01
        self.max_tr_rad = 1
        self.tr_succ_thresh = 5
        self.tr_fail_thresh = 5

        self.tr_rad = 0.25
        self.tr_succ_count = 0
        self.tr_fail_count = 0
    
    def expected_improvement(self, x, gp, y_best):
        mean, std = gp.predict(x, return_std=True)
        with np.errstate(divide='warn'):
            improvement = y_best - mean
            z = improvement / std
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei[std == 0.0] = 0.0  
        return ei
    
    def run(self, budget):
        while budget > 0:
            best_x = self.obs_x[min(range(len(self.obs_y)), key=lambda i: self.obs_y[i])]
            tr_obs = [(obs_x, obs_y) for obs_x, obs_y in zip(self.obs_x, self.obs_y) if np.max(np.abs(obs_x - best_x)) <= self.tr_rad]
            self.gp.fit(*zip(*tr_obs))

            proxy_sample_x = random_sample(self.d, self.proxy_sample_size, origin=best_x, radius=self.tr_rad)
            y_best = min(self.obs_y)
            ei_values = self.expected_improvement(proxy_sample_x, self.gp, y_best)
            best_proxy_x = proxy_sample_x[np.argmax(ei_values)]

            self.evaluate(best_proxy_x)
            budget -= 1
            improvement = self.best_y[-1] > self.best_y[-2]
            
            if improvement:
                self.tr_succ_count += 1
            else:
                self.tr_fail_count += 1

            if self.tr_succ_count >= self.tr_succ_thresh:
                self.tr_rad = max(2 * self.tr_rad, self.max_tr_rad)
            elif self.tr_fail_count >= self.tr_fail_thresh:
                self.tr_rad /= 2
                if self.tr_rad < self.min_tr_rad:
                    doe_budget = min(self.doe_sample_budget, budget)
                    self.__init__(self.problem, self.d, doe_budget, is_reset=True)
                    budget -= doe_budget

            if self.tr_succ_count >= self.tr_succ_thresh or self.tr_fail_count >= self.tr_fail_thresh:
                self.tr_succ_count, self.tr_fail_count = 0, 0




if __name__ == "__main__":
    # Hyperparameter tuning to determine the best parameters for both problems
    #population_size, mutation_rate, crossover_rate = 
    population_size, mutation_rate, crossover_rate = tune_hyperparameters()
    print(population_size)
    print(mutation_rate)
    print(crossover_rate)