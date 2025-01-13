from typing import List
import numpy as np
from scipy.stats import norm
from scipy.stats import qmc
import sklearn.gaussian_process
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
from GA import studentnumber1_studentnumber2_GA, create_problem
budget = 1000000
# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`
seed = 42
np.random.seed(42)


d = 3  
num = 5 
origin = [105, 0.05, 0.6] 
radius = [95, 0.05, 0.35]

# Hyperparameter tuning function
def tune_hyperparameters() -> List:
    # You should decide/engineer the `score` youself, which is the tuning objective
    # create the LABS problem and the data logger
    F18, _logger_18 = create_problem(dimension=50, fid=18)
    # create the N-Queens problem and the data logger
    F23, _logger_23 = create_problem(dimension=49, fid=23)
    
    run_budget = 20
    
    # ********** optimization ***********
    sample_budget = 20
    '''sampler_1 = RandomSampler(GA_evaluation, d, sample_budget, F18, F23)
    sampler_1.run(budget=run_budget)'''

    '''sampler_2 = GlobalBO(GA_evaluation, d, sample_budget, F18, F23)
    sampler_2.run(budget=run_budget)'''

    '''sampler_3 = GlobalBO_EI(GA_evaluation, d, sample_budget, F18, F23)
    sampler_3.run(budget=run_budget)'''

    '''sampler_4 = TuRBO(GA_evaluation, d, sample_budget, F18, F23)
    sampler_4.run(budget=run_budget)'''

    sampler_5 = TuRBO_EI(GA_evaluation, d, sample_budget, F18, F23)
    sampler_5.run(budget=run_budget)

    
    pop_size, mutation_rate, crossover_rate = sampler_5.best_x[-1]
    
    pop_size = round(pop_size)
    mutation_rate = round(mutation_rate, 2)
    crossover_rate = round(crossover_rate, 1)
    return pop_size, mutation_rate, crossover_rate


def GA_evaluation(problem1, problem2, sample, runs=5):
    pop_size, mutation_rate, crossover_rate = sample
    pop_size = round(pop_size)
    F18 = []
    F23 = []
    for run in range(runs):
        studentnumber1_studentnumber2_GA(problem1, pop_size, mutation_rate, crossover_rate)
        # print(problem1.state.current_best)
        F_18 = problem1.state.current_best.y
        F18.append(F_18)
        problem1.reset()
        studentnumber1_studentnumber2_GA(problem2, pop_size, mutation_rate, crossover_rate)
        F_23 = problem2.state.current_best.y
        F23.append(F_23)
        # print(problem2.state.current_best)
        problem2.reset()
    # ********** set different weights after recording the initial results *******
    score = balanced_weights(F18, F23)
    return score


# ********* weight *****************
def dynamic_weights(F18, F23, epsilon=1e-6):
    F18_max = max(F18)
    F23_max = max(F23)
    w1 = 1 / (abs(F18_max) + epsilon)
    w2 = 1 / (abs(F23_max) + epsilon)
    total_weight = w1 + w2
    w1 /= total_weight
    w2 /= total_weight
    score = w1*F18_max + F23_max*w2
    return score

def balanced_weights(F18, F23, epsilon=1e-6):
    F18_min, F18_max = min(F18), max(F18)
    F23_min, F23_max = min(F23), max(F23)
    delta_F18 = F18_max - F18_min + epsilon
    delta_F23 = F23_max - F23_min + epsilon
    total = delta_F18 + delta_F23
    w1 = delta_F18 / total
    w2 = delta_F23 / total
    score = w1 * F18_max**2 + F23_max*w2
    return score

def balanced_weights_bias(F18, F23, alpha=1.05, epsilon=1e-6):
    F18_min, F18_max = min(F18), max(F18)
    F23_min, F23_max = min(F23), max(F23)
    delta_F18 = F18_max - F18_min + epsilon
    delta_F23 = F23_max - F23_min + epsilon
    total = alpha * delta_F18 + delta_F23
    w1 = (alpha * delta_F18) / total
    w2 = delta_F23 / total
    score = w1 * F18_max + w2 * F23_max
    return score


def normalized_weights(F18, F23, epsilon=1e-6):
    epsilon = 1e-6
    F18_min, F18_max = min(F18), max(F18)
    F23_min, F23_max = min(F23), max(F23)
    norm_F18 = [(f - F18_min) / (F18_max - F18_min + epsilon) for f in F18]
    norm_F23 = [(f - F23_min) / (F23_max - F23_min + epsilon) for f in F23]
    norm_sum = sum(norm_F18) + sum(norm_F23)
    w1 = sum(norm_F18) / norm_sum
    w2 = sum(norm_F23) / norm_sum
    score = w1*F18_max + F23_max*w2
    return score

def adaptive_weights(F18, F23, alpha=0.5, epsilon=1e-6):
    F18_min, F18_max = min(F18), max(F18)
    F23_min, F23_max = min(F23), max(F23)
    delta_F18 = F18_max - F18_min + epsilon
    delta_F23 = F23_max - F23_min + epsilon
    total = delta_F18 + delta_F23
    w1 = alpha * (delta_F18 / total)
    w2 = (1 - alpha) * (delta_F23 / total)
    score = w1*F18_max + F23_max*w2
    return score



def random_sample(d, num, origin=origin, radius=radius):
    origin = np.array(origin)
    radius = np.array(radius)
    # Latin Hypercube Sampling (LHS) samples
    if np.any(origin < 0):
        origin = np.maximum(origin, 0)
    
    min_value = origin - radius
    min_value = np.maximum(min_value, 0) 
    max_value = origin + radius

    sampler = qmc.LatinHypercube(d, seed=seed)
    raw_samples = sampler.random(num) 
    scaled_samples = qmc.scale(raw_samples, min_value, max_value)
    scaled_samples[:, 0] = np.round(scaled_samples[:, 0]).astype(int)
    scaled_samples[:, 0] = np.clip(scaled_samples[:, 0], min_value[0], max_value[0])
    scaled_samples[:, 0] = np.maximum(scaled_samples[:, 0], 3)
    
    scaled_samples[:, 1] = np.clip(scaled_samples[:, 1], 0.01, 0.1)
    scaled_samples[:, 2] = np.clip(scaled_samples[:, 2], 0.1, 0.9)

    return scaled_samples


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
        
        # kernel = sklearn.gaussian_process.kernels.Matern()
        kernel = None
        self.gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, random_state=seed)
        self.proxy_sample_size = 10

    def run(self, budget):
        for _ in range(budget):
            # fit GP model
            self.gp.fit(self.obs_x, self.obs_y)

            # Thompson sample
            proxy_sample_x = random_sample(self.d, self.proxy_sample_size)
            proxy_sample_y = self.gp.sample_y(proxy_sample_x).reshape(-1)
            best_proxy_x = proxy_sample_x[max(range(self.proxy_sample_size), key=lambda i: proxy_sample_y[i])]

            self.evaluate(best_proxy_x)

class GlobalBO_EI(RandomSampler):
    def __init__(self, problem, d, doe_sample_budget, *problem_args, **kwargs):
        super().__init__(problem, d, doe_sample_budget, *problem_args, **kwargs)
        
        kernel = None
        self.gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, random_state=seed)
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
            y_best = max(self.obs_y)
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
            best_x = self.obs_x[max(range(len(self.obs_y)), key=lambda i: self.obs_y[i])]
            tr_obs = [(obs_x, obs_y) for obs_x, obs_y in zip(self.obs_x, self.obs_y) if np.max(np.abs(obs_x - best_x)) <= self.tr_rad]
            self.gp.fit(*zip(*tr_obs))

            proxy_sample_x = random_sample(self.d, self.proxy_sample_size, origin=best_x, radius=self.tr_rad)

            # Thompson sampling
            proxy_sample_y = self.gp.sample_y(proxy_sample_x).reshape(-1)
            best_proxy_x = proxy_sample_x[max(range(self.proxy_sample_size), key=lambda i: proxy_sample_y[i])]
            
            # if np.any(best_proxy_x) <=0 :
            
            self.evaluate(best_proxy_x)
            budget -= 1
            improvement = None
            if len(self.best_y)>1:
                improvement = self.best_y[-1] > 5
                
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
            best_x = self.obs_x[max(range(len(self.obs_y)), key=lambda i: self.obs_y[i])]
            tr_obs = [(obs_x, obs_y) for obs_x, obs_y in zip(self.obs_x, self.obs_y) if np.max(np.abs(obs_x - best_x)) <= self.tr_rad]
            self.gp.fit(*zip(*tr_obs))

            proxy_sample_x = random_sample(self.d, self.proxy_sample_size, origin=best_x, radius=self.tr_rad)
            y_best = max(self.obs_y)
            ei_values = self.expected_improvement(proxy_sample_x, self.gp, y_best)
            best_proxy_x = proxy_sample_x[np.argmax(ei_values)]
            print(best_proxy_x)
            self.evaluate(best_proxy_x)
            budget -= 1
            improvement = None
            if len(self.best_y)>1:
                improvement = self.best_y[-1] > 5
                if improvement is True:
                    print(self.best_y)
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

            print(f"improvement:{improvement} ")


if __name__ == "__main__":
    # Hyperparameter tuning to determine the best parameters for both problems 
    population_size, mutation_rate, crossover_rate = tune_hyperparameters()
    print(population_size)
    print(mutation_rate)
    print(crossover_rate)
    