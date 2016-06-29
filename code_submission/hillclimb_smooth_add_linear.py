import cvxpy
import scipy as sp
import numpy as np
from convexopt_solvers import SmoothAndLinearProblemWrapper
from data_generation import smooth_plus_linear
from common import *

NUMBER_OF_ITERATIONS = 60
BOUNDARY_FACTOR = 0.8
STEP_SIZE = 0.5
LAMBDA_MIN = 1e-6
SHRINK_MIN = 1e-15
SHRINK_SHRINK_FACTOR = 0.1
SHRINK_FACTOR_INIT = 1
DECREASING_ENOUGH_THRESHOLD = 1e-4
BACKTRACK_ALPHA = 0.01

def _get_order_indices(Xs_train, Xs_validate, Xs_test):
    Xs = np.vstack((Xs_train, Xs_validate, Xs_test))
    order_indices = np.argsort(np.reshape(np.array(Xs), Xs.size), axis=0)
    num_train = Xs_train.shape[0]
    num_train_and_validate = num_train + Xs_validate.shape[0]
    train_indices = np.reshape(np.array(np.less(order_indices, num_train)), order_indices.size)
    validate_indices = np.logical_and(
        np.logical_not(train_indices),
        np.reshape(np.array(np.less(order_indices, num_train_and_validate)), order_indices.size)
    )
    return Xs[order_indices], order_indices, train_indices, validate_indices

def _get_reordered_data(train_data, validate_data, order_indices, train_indices, validate_indices):
    num_features = train_data.shape[1]
    dummy_data = np.zeros((TEST_SIZE, num_features))
    combined_data = np.vstack((train_data, validate_data, dummy_data))
    ordered_data = combined_data[order_indices]
    return ordered_data[train_indices], ordered_data[validate_indices]

def run(Xl_train, Xs_train, y_train, Xl_validate, Xs_validate, y_validate, Xs_test, initial_lambdas=[]):
    method_label = "HCSmoothLinear2"

    # We need to reorder all the data (y, X_linear, and X_smooth) by increasing X_smooth
    # combine Xs_train and Xs_validate and sort
    Xs_ordered, order_indices, train_indices, validate_indices = _get_order_indices(Xs_train, Xs_validate, Xs_test)
    Xl_train_ordered, Xl_validate_ordered = _get_reordered_data(Xl_train, Xl_validate, order_indices, train_indices, validate_indices)
    y_train_ordered, y_validate_ordered = _get_reordered_data(y_train, y_validate, order_indices, train_indices, validate_indices)

    curr_regularization = np.array(initial_lambdas)
    first_regularization = curr_regularization

    problem_wrapper = SmoothAndLinearProblemWrapper(Xl_train_ordered, Xs_ordered, train_indices, y_train_ordered)
    difference_matrix = problem_wrapper.D
    beta, thetas = problem_wrapper.solve(curr_regularization)
    if beta is None or thetas is None:
        # there was a problem finding a solution
        return beta, thetas, [], curr_regularization

    current_cost = testerror_smooth_and_linear(Xl_validate_ordered, y_validate_ordered, beta, thetas[validate_indices])

    # track progression
    cost_path = [current_cost]

    method_step_size = STEP_SIZE
    shrink_factor = SHRINK_FACTOR_INIT
    potential_thetas = None
    potential_betas = None
    for i in range(0, NUMBER_OF_ITERATIONS):
        print "ITER", i, "current lambdas", curr_regularization

        try:
            lambda_derivatives = _get_lambda_derivatives(Xl_train_ordered, y_train_ordered, Xl_validate_ordered, y_validate_ordered, beta, thetas, train_indices, validate_indices, difference_matrix, curr_regularization)
        except np.linalg.LinAlgError as e:
            # linalg error
            break

        if np.any(np.isnan(lambda_derivatives)):
            # trouble calculating derivatives
            break

        potential_new_regularization = _get_updated_lambdas(curr_regularization, shrink_factor * method_step_size, lambda_derivatives, boundary_factor=BOUNDARY_FACTOR)
        try:
            potential_beta, potential_thetas = problem_wrapper.solve(potential_new_regularization)
        except cvxpy.error.SolverError:
            potential_beta = None
            potential_thetas = None

        if potential_beta is None or potential_thetas is None:
            # cvxpy could not find a soln
            potential_cost = current_cost * 100
        else:
            potential_cost = testerror_smooth_and_linear(Xl_validate_ordered, y_validate_ordered, potential_beta, potential_thetas[validate_indices])

        raw_backtrack_check = current_cost - BACKTRACK_ALPHA * shrink_factor * method_step_size * np.linalg.norm(lambda_derivatives)**2
        backtrack_check = current_cost if raw_backtrack_check < 0 else raw_backtrack_check
        while potential_cost > backtrack_check and shrink_factor > SHRINK_MIN:
            # shrink step size according to backtracking descent
            shrink_factor *= SHRINK_SHRINK_FACTOR

            potential_new_regularization = _get_updated_lambdas(curr_regularization, shrink_factor * method_step_size, lambda_derivatives, boundary_factor=BOUNDARY_FACTOR)
            try:
                potential_beta, potential_thetas = problem_wrapper.solve(potential_new_regularization)
            except cvxpy.error.SolverError as e:
                # cvxpy could not find a soln
                potential_beta = None
                potential_thetas = None

            if potential_beta is None or potential_thetas is None:
                potential_cost = current_cost * 100
            else:
                potential_cost = testerror_smooth_and_linear(Xl_validate_ordered, y_validate_ordered, potential_beta, potential_thetas[validate_indices])

            raw_backtrack_check = current_cost - BACKTRACK_ALPHA * shrink_factor * method_step_size * np.linalg.norm(lambda_derivatives)**2
            backtrack_check = current_cost if raw_backtrack_check < 0 else raw_backtrack_check

        # track progression
        if cost_path[-1] < potential_cost:
            # cost increasing
            break
        else:
            curr_regularization = potential_new_regularization
            current_cost = potential_cost
            beta = potential_beta
            thetas = potential_thetas
            cost_path.append(current_cost)

            if cost_path[-2] - cost_path[-1] < DECREASING_ENOUGH_THRESHOLD:
                # progress too slow
                break

        if shrink_factor < SHRINK_MIN:
            # shrink size too small
            break

    return beta, thetas, cost_path, curr_regularization

def _get_updated_lambdas(lambdas, method_step_size, lambda_derivatives, boundary_factor=None):
    new_step_size = method_step_size
    if boundary_factor is not None:
        # do not let lambda values get too small
        potential_lambdas = lambdas - method_step_size * lambda_derivatives

        for idx in range(0, lambdas.size):
            if lambdas[idx] > LAMBDA_MIN and potential_lambdas[idx] < (1 - boundary_factor) * lambdas[idx]:
                smaller_step_size = boundary_factor * lambdas[idx] / lambda_derivatives[idx]
                new_step_size = min(new_step_size, smaller_step_size)

    return np.maximum(lambdas - new_step_size * lambda_derivatives, LAMBDA_MIN)

def _get_lambda_derivatives(Xl_train, y_train, Xl_validate, y_validate, beta, thetas, train_indices, validate_indices, difference_matrix, lambdas):
    def _get_zero_or_sign_vector(vector):
        zero_indices = np.logical_not(get_nonzero_indices(vector, threshold=1e-7))
        vector_copy = np.sign(vector)
        vector_copy[zero_indices] = 0
        return vector_copy

    dbeta_dlambdas = [0] * len(lambdas)
    dtheta_dlambdas = [0] * len(lambdas)
    num_samples = thetas.size
    num_beta_features = beta.size

    eye_matrix_features = np.matrix(np.identity(num_beta_features))
    M = np.matrix(np.identity(num_samples))[train_indices, :]
    MM = M.T * M
    XX = Xl_train.T * Xl_train
    DD = difference_matrix.T * difference_matrix

    inv_matrix12 = sp.linalg.pinvh(MM + lambdas[2] * DD)
    inv_matrix12_X = inv_matrix12 * M.T * Xl_train
    matrix12_to_inv = XX + lambdas[1] * eye_matrix_features - Xl_train.T * M * inv_matrix12_X
    dbeta_dlambdas[0], _, _, _ = np.linalg.lstsq(matrix12_to_inv, -1 * _get_zero_or_sign_vector(beta))
    dtheta_dlambdas[0] = -1 * inv_matrix12_X * dbeta_dlambdas[0]

    dbeta_dlambdas[1], _, _, _ = np.linalg.lstsq(matrix12_to_inv, -1 * beta)
    dtheta_dlambdas[1] = -1 * inv_matrix12_X * dbeta_dlambdas[1]

    inv_matrix34 = sp.linalg.pinvh(XX + lambdas[1] * eye_matrix_features)
    inv_matrix34_X = inv_matrix34 * Xl_train.T * M
    matrix34_to_inv = MM + lambdas[2] * DD - M.T * Xl_train * inv_matrix34_X

    dtheta_dlambdas[2], _, _, _ = np.linalg.lstsq(matrix34_to_inv, -1.0 * difference_matrix.T * difference_matrix * thetas)
    dbeta_dlambdas[2] = -1 * inv_matrix34_X * dtheta_dlambdas[2]
    err_vector = y_validate - Xl_validate * beta - thetas[validate_indices]

    df_dlambdas = [
        -1 * ((Xl_validate * dbeta_dlambdas[i] + dtheta_dlambdas[i][validate_indices]).T * err_vector)[0,0]
        for i in range(0, len(lambdas))
    ]

    return np.array(df_dlambdas)
