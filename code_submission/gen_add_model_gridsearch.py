import sys
import itertools
import numpy as np

from common import *
from convexopt_solvers import GenAddModelProblemWrapper

LAMBDA_MIN_FACTOR = 1e-3

def run(y_train, y_validate, X_full, train_idx, validate_idx, num_lambdas=10, max_lambda=100):
    max_power = np.log(max_lambda)
    min_power = np.log(LAMBDA_MIN_FACTOR * max_power)
    lambda_guesses = np.power(np.e, np.arange(min_power, max_power, (max_power - min_power - 0.01) / (num_lambdas - 1)))
    print "gridsearch: lambda_guesses", lambda_guesses

    num_features = X_full.shape[1]
    num_samples = X_full.shape[0]
    problem_wrapper = GenAddModelProblemWrapper(X_full, train_idx, y_train)

    best_cost = 1e5
    best_thetas = []
    best_regularization = [lambda_guesses[0]] * num_features

    all_lambda_combos = list()
    if num_features <= 2:
        all_regs = itertools.product(*([lambda_guesses] * num_features))
    else:
        all_regs = [[l] * num_features for l in lambda_guesses]

    for regularization in all_regs:
        print "gridsearch regularization", regularization
        thetas = problem_wrapper.solve(regularization, high_accur=False)
        if thetas is not None:
            current_cost = testerror_multi_smooth(y_validate, validate_idx, thetas)
            if best_cost > current_cost:
                best_cost = current_cost
                best_thetas = thetas
                best_regularization = regularization
                print "gridsearch: cost", best_cost, "regularization", best_regularization

    print "gridsearch: best_validation_error", best_cost
    print "gridsearch: best lambdas:", best_regularization

    return best_thetas, best_cost
