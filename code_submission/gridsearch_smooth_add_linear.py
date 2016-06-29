import cvxpy
import scipy as sp
import numpy as np
from common import *
from convexopt_solvers import SmoothAndLinearProblemWrapperSimple

DEFAULT_NUM_LAMBDAS = 10
MIN_POWER = np.log(1e-6)
MAX_POWER = np.log(1e1)

# Xl_* is the linear covariates, Xs_* is the smooth covariates
def run(Xl_train, Xs_train, y_train, Xl_validate, Xs_validate, y_validate, Xs_test, num_lambdas=DEFAULT_NUM_LAMBDAS):
    # lambdas to run grid search over
    lambda_guesses = np.power(np.e, np.arange(MIN_POWER, MAX_POWER, (MAX_POWER - MIN_POWER - 0.1) / num_lambdas))

    # Must order the smooth data
    # Combine the data and then generate indices for which dataset the observations belong to
    Xs = np.vstack((Xs_train, Xs_validate, Xs_test))
    order_indices = np.argsort(np.reshape(np.array(Xs), Xs.size), axis=0)

    num_train = Xs_train.shape[0]
    num_train_and_validate = num_train + Xs_validate.shape[0]
    train_indices = np.reshape(np.array(np.less(order_indices, num_train)), order_indices.size)
    validate_indices = np.logical_and(
        np.logical_not(train_indices),
        np.reshape(np.array(np.less(order_indices, num_train_and_validate)), order_indices.size)
    )
    Xs_ordered = Xs[order_indices]

    def _get_reordered_data(train_data, validate_data):
        dummy_data = np.zeros((TEST_SIZE, train_data.shape[1]))
        combined_data = np.concatenate((train_data, validate_data, dummy_data))
        ordered_data = combined_data[order_indices]
        return ordered_data[train_indices], ordered_data[validate_indices]

    # Reorder the rest of the data to match the order of the smooth data
    Xl_train_ordered, Xl_validate_ordered = _get_reordered_data(Xl_train, Xl_validate)
    y_train_ordered, y_validate_ordered = _get_reordered_data(y_train, y_validate)

    problem_wrapper = SmoothAndLinearProblemWrapperSimple(Xl_train_ordered, Xs_ordered, train_indices, y_train_ordered)

    best_beta = []
    best_thetas = []
    best_cost = np.inf
    best_lambdas = [lambda_guesses[0]] * 2

    for l1 in lambda_guesses:
        for l2 in lambda_guesses:
            try:
                beta, thetas = problem_wrapper.solve([l1, l2])
            except cvxpy.error.SolverError:
                continue
            if beta is None or thetas is None:
                continue
            current_cost = testerror_smooth_and_linear(Xl_validate_ordered, y_validate_ordered, beta, thetas[validate_indices])

            if best_cost > current_cost:
                best_cost = current_cost
                best_beta = beta
                best_thetas = thetas
                best_lambdas = [l1, l2]

    return best_beta, best_thetas, best_cost
