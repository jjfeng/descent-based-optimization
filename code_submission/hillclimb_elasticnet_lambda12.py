import numpy as np
import scipy as sp
from common import *
from convexopt_solvers import Lambda12ProblemWrapper

NUMBER_OF_ITERATIONS = 60

STEP_SIZE = 0.4
MIN_LAMBDA = 1e-10
BOUNDARY_FACTOR = 0.7
MIN_SHRINK = 1e-8
SHRINK_INIT = 1
SHRINK_SHRINK = 0.05
DECREASING_ENOUGH_THRESHOLD = 1e-4
BACKTRACK_ALPHA = 0.01

def run(X_train, y_train, X_validate, y_validate, initial_lambda1=1, initial_lambda2=1):
    shrink_factor = SHRINK_INIT
    method_step_size = STEP_SIZE

    # our guesses for the params
    curr_lambdas = [initial_lambda1, initial_lambda2]
    problem_wrapper = Lambda12ProblemWrapper(X_train, y_train)
    beta_guess = problem_wrapper.solve(curr_lambdas[0], curr_lambdas[1])

    current_cost = testerror(X_validate, y_validate, beta_guess)
    cost_path = [current_cost]
    for i in range(0, NUMBER_OF_ITERATIONS):
        print "ITER", i, "current lambdas", curr_lambdas
        prev_lambdas = curr_lambdas

        derivative_lambdas = _get_derivative_lambda(X_train, y_train, X_validate, y_validate, beta_guess, curr_lambdas[1])
        potential_lambdas = _get_updated_lambdas(curr_lambdas, method_step_size * shrink_factor, derivative_lambdas, use_boundary=False)

        # get corresponding beta
        potential_beta_guess = problem_wrapper.solve(potential_lambdas[0], potential_lambdas[1])
        potential_cost = testerror(X_validate, y_validate, potential_beta_guess)

        # Do backtracking line descent if needed
        while potential_cost > current_cost - BACKTRACK_ALPHA * shrink_factor * method_step_size * np.linalg.norm(derivative_lambdas)**2 and shrink_factor > MIN_SHRINK:
            shrink_factor *= SHRINK_SHRINK
            potential_lambdas = _get_updated_lambdas(curr_lambdas, method_step_size * shrink_factor, derivative_lambdas)
            potential_beta_guess = problem_wrapper.solve(potential_lambdas[0], potential_lambdas[1])
            potential_cost = testerror(X_validate, y_validate, potential_beta_guess)

        delta_change = current_cost - potential_cost
        if potential_cost < current_cost:
            current_cost = potential_cost
            beta_guess = potential_beta_guess
            curr_lambdas = potential_lambdas

        if shrink_factor <= MIN_SHRINK:
            break

        if curr_lambdas[0] == prev_lambdas[0] and curr_lambdas[1] == prev_lambdas[1]:
            break

        if delta_change <= DECREASING_ENOUGH_THRESHOLD:
            break

        cost_path.append(current_cost)

    return beta_guess, cost_path

def _get_updated_lambdas(current_lambdas, step_size, derivative_lambdas, use_boundary=False):
    derivatives = np.array(derivative_lambdas)
    lambdas = np.array(current_lambdas)
    potential_lambdas = lambdas - step_size * derivatives

    new_step_size = step_size
    if use_boundary:
        # Do not get lambdas get too small
        for idx in range(0, len(lambdas)):
            if potential_lambdas[idx] < (1 - BOUNDARY_FACTOR) * lambdas[idx]:
                smaller_step_size = BOUNDARY_FACTOR * lambdas[idx] / derivatives[idx]
                new_step_size = min(new_step_size, smaller_step_size)

        return lambdas - new_step_size * derivatives
    else:
        return np.maximum(MIN_LAMBDA, lambdas - new_step_size * derivatives)

def _get_derivative_lambda(X_train, y_train, X_validate, y_validate, beta_guess, lambda2):
    # only need the current value of lambda2 (the lambda for the L2-norm) to get the derivative
    nonzero_indices = get_nonzero_indices(beta_guess)

    # If everything is zero, gradient is zero
    if np.sum(nonzero_indices) == 0:
        return [0, 0]

    X_train_mini = X_train[:, nonzero_indices]
    X_validate_mini = X_validate[:, nonzero_indices]
    beta_guess_mini = beta_guess[nonzero_indices]

    eye_matrix = np.matrix(np.identity(beta_guess_mini.size))
    to_invert_matrix = X_train_mini.T * X_train_mini + lambda2 * eye_matrix

    dbeta_dlambda1, _, _, _ = np.linalg.lstsq(to_invert_matrix, -1 * np.sign(beta_guess_mini))
    dbeta_dlambda2, _, _, _ = np.linalg.lstsq(to_invert_matrix, -1 * beta_guess_mini)

    err_vector = y_validate - X_validate_mini * beta_guess_mini
    gradient_lambda1 = -1 * (X_validate_mini * dbeta_dlambda1).T * err_vector
    gradient_lambda2 = -1 * (X_validate_mini * dbeta_dlambda2).T * err_vector

    return [gradient_lambda1[0,0], gradient_lambda2[0,0]]
