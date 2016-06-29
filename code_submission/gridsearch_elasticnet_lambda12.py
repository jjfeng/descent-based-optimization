import numpy as np

from common import *
from convexopt_solvers import Lambda12ProblemWrapper

LAMBDA_EIGEN_FACTOR = 4
NUM_LAMBDAS = 10

def run(X_train, y_train, X_validate, y_validate):
    largest_eigenvalue = max(np.linalg.eigvalsh(X_train.T * X_train))
    max_power = np.log(largest_eigenvalue * LAMBDA_EIGEN_FACTOR)
    min_power = np.log(1e-5)
    lambda_guesses = np.power(np.e, np.arange(min_power, max_power, (max_power - min_power - 0.01) / NUM_LAMBDAS))
    print "lambda_guesses", lambda_guesses

    # Create the problem - allows for warmstart
    problem_wrapper = Lambda12ProblemWrapper(X_train, y_train)

    best_beta = []   # initialize
    best_cost = 1e5  # initialize to something huge
    best_lambda1 = lambda_guesses[0]
    best_lambda2 = lambda_guesses[0]

    for l1 in lambda_guesses:
        for l2 in lambda_guesses:
            beta_guess = problem_wrapper.solve(l1, l2)
            current_cost = testerror(X_validate, y_validate, beta_guess)

            if best_cost > current_cost:
                best_cost = current_cost
                best_beta = beta_guess
                best_lambda1 = l1
                best_lambda2 = l2

    print "gridsearch: best_validation_error", best_cost
    print "gridsearch: best lambdas:", (best_lambda1, best_lambda2)

    return best_beta
