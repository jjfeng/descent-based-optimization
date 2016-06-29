import matplotlib.pyplot as plt
from data_generation import smooth_plus_linear
import cvxpy
import scipy as sp
import numpy as np
from common import *
from convexopt_solvers import SmoothAndLinearProblemWrapperSimple

NUMBER_OF_ITERATIONS = 40
BOUNDARY_FACTOR = 0.5
STEP_SIZE = 0.5
MAX_STEP_SIZE = 1
LAMBDA_MIN = 1e-10
SHRINK_MIN = 1e-15
METHOD_STEP_SIZE_MIN = 1e-32
SHRINK_SHRINK_FACTOR = 0.1
SHRINK_FACTOR_INIT = 1
DECREASING_ENOUGH_THRESHOLD = 3 * 1e-4
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

def run(Xl_train, Xs_train, y_train, Xl_validate, Xs_validate, y_validate, Xs_test, initial_lambdas=[1,1]):
    method_label = "HCSmoothLinearSimple"

    # We need to reorder all the data (y, X_linear, and X_smooth) by increasing X_smooth
    # combine Xs_train and Xs_validate and sort
    Xs_ordered, order_indices, train_indices, validate_indices = _get_order_indices(Xs_train, Xs_validate, Xs_test)
    Xl_train_ordered, Xl_validate_ordered = _get_reordered_data(Xl_train, Xl_validate, order_indices, train_indices, validate_indices)
    y_train_ordered, y_validate_ordered = _get_reordered_data(y_train, y_validate, order_indices, train_indices, validate_indices)

    curr_regularization = np.array(initial_lambdas)
    first_regularization = curr_regularization

    problem_wrapper = SmoothAndLinearProblemWrapperSimple(Xl_train_ordered, Xs_ordered, train_indices, y_train_ordered, use_l1=False)
    difference_matrix = problem_wrapper.D
    beta, thetas = problem_wrapper.solve(curr_regularization)
    if beta is None or thetas is None:
        return beta, thetas, [], curr_regularization

    current_cost = testerror_smooth_and_linear(Xl_validate_ordered, y_validate_ordered, beta, thetas[validate_indices])
    print method_label, "first_regularization", first_regularization, "first cost", current_cost

    # track progression
    cost_path = [current_cost]

    method_step_size = STEP_SIZE
    shrink_factor = SHRINK_FACTOR_INIT
    potential_thetas = None
    potential_betas = None
    for i in range(0, NUMBER_OF_ITERATIONS):
        shrink_factor = min(MAX_STEP_SIZE, shrink_factor * 2)
        try:
            lambda_derivatives = _get_lambda_derivatives(Xl_train_ordered, y_train_ordered, Xl_validate_ordered, y_validate_ordered, beta, thetas, train_indices, validate_indices, difference_matrix, curr_regularization)
        except np.linalg.LinAlgError as e:
            print "linalg error. returning early", e
            break

        if np.any(np.isnan(lambda_derivatives)):
            print "some value in df_dlambda is nan"
            break

        potential_new_regularization = _get_updated_lambdas(curr_regularization, shrink_factor * method_step_size, lambda_derivatives, boundary_factor=0.9)
        try:
            potential_beta, potential_thetas = problem_wrapper.solve(potential_new_regularization)
        except cvxpy.error.SolverError:
            potential_beta = None
            potential_thetas = None

        if potential_beta is None or potential_thetas is None:
            print "cvxpy could not find a soln"
            potential_cost = current_cost * 100
        else:
            potential_cost = testerror_smooth_and_linear(Xl_validate_ordered, y_validate_ordered, potential_beta, potential_thetas[validate_indices])

        while potential_cost >= current_cost - BACKTRACK_ALPHA * shrink_factor * method_step_size * np.linalg.norm(lambda_derivatives)**2 and shrink_factor > SHRINK_MIN:
            if potential_cost > 2 * current_cost:
                shrink_factor *= SHRINK_SHRINK_FACTOR * 0.01
            else:
                shrink_factor *= SHRINK_SHRINK_FACTOR

            potential_new_regularization = _get_updated_lambdas(curr_regularization, shrink_factor * method_step_size, lambda_derivatives, boundary_factor=0.9)
            print "potential_new_regularization", potential_new_regularization
            try:
                potential_beta, potential_thetas = problem_wrapper.solve(potential_new_regularization)
            except cvxpy.error.SolverError as e:
                print "cvxpy could not find a soln", e
                potential_beta = None
                potential_thetas = None

            if potential_beta is None or potential_thetas is None:
                potential_cost = current_cost * 100
                print "try shrink", shrink_factor, "no soln. oops!"
            else:
                potential_cost = testerror_smooth_and_linear(Xl_validate_ordered, y_validate_ordered, potential_beta, potential_thetas[validate_indices])
                print "try shrink", shrink_factor, "potential_cost", potential_cost

        # track progression
        if cost_path[-1] < potential_cost:
            print "COST IS INCREASING!"
            break
        else:
            curr_regularization = potential_new_regularization
            current_cost = potential_cost
            beta = potential_beta
            thetas = potential_thetas
            cost_path.append(current_cost)

            print method_label, "iter:", i, "current_cost:", current_cost, "lambdas:", curr_regularization, "shrink_factor", shrink_factor

            if cost_path[-2] - cost_path[-1] < DECREASING_ENOUGH_THRESHOLD:
                print "progress too slow", cost_path[-2] - cost_path[-1]
                break

        if shrink_factor < SHRINK_MIN:
            print method_label, "SHRINK SIZE TOO SMALL", "shrink_factor", shrink_factor
            break

    print method_label, "current cost", current_cost, "curr_regularization", curr_regularization, "total iters:", i
    print method_label, "curr lambdas:",  "first cost", cost_path[0], "first_regularization", first_regularization

    return beta, thetas, cost_path, curr_regularization


def run_nesterov(Xl_train, Xs_train, y_train, Xl_validate, Xs_validate, y_validate, Xs_test, initial_lambda1=1, initial_lambda2=1):
    def _get_beta_theta_cost(problem_wrapper, Xl_validate_ordered, y_validate_ordered, regularization):
        try:
            potential_beta, potential_thetas = problem_wrapper.solve(regularization)
        except cvxpy.error.SolverError:
            print "cvxpy could not find a soln."
            potential_beta = None
            potential_thetas = None

        if potential_beta is not None and potential_thetas is not None:
            potential_cost = testerror_smooth_and_linear(Xl_validate_ordered, y_validate_ordered, potential_beta, potential_thetas[validate_indices])
        else:
            potential_cost = 1e10

        return potential_beta, potential_thetas, potential_cost

    def _get_accelerated_lambdas(curr_lambdas, prev_lambdas, iter_num):
        return np.maximum(
            curr_lambdas + (iter_num - 2) / (iter_num + 1.0) * (curr_lambdas - prev_lambdas),
            np.minimum(curr_lambdas, LAMBDA_MIN)
        )
    potential_beta = None
    potential_theta = None
    method_label = "HCSmoothLinear Nesterov Simple"

    # We need to reorder all the data (y, X_linear, and X_smooth) by increasing X_smooth
    # combine Xs_train and Xs_validate and sort
    Xs_ordered, order_indices, train_indices, validate_indices = _get_order_indices(Xs_train, Xs_validate, Xs_test)
    Xl_train_ordered, Xl_validate_ordered = _get_reordered_data(Xl_train, Xl_validate, order_indices, train_indices, validate_indices)
    y_train_ordered, y_validate_ordered = _get_reordered_data(y_train, y_validate, order_indices, train_indices, validate_indices)

    curr_regularization = np.array([initial_lambda1, initial_lambda2])
    prev_regularization = curr_regularization
    acc_regularization = curr_regularization
    problem_wrapper = SmoothAndLinearProblemWrapperSimple(Xl_train_ordered, Xs_ordered, train_indices, y_train_ordered, use_l1=False)
    difference_matrix = problem_wrapper.D
    beta, thetas = problem_wrapper.solve(acc_regularization)

    current_cost = testerror_smooth_and_linear(Xl_validate_ordered, y_validate_ordered, beta, thetas[validate_indices])
    first_regularization = curr_regularization
    print "first_regularization", first_regularization, "first cost", current_cost

    # track progression
    cost_path = [current_cost]

    lambda_derivatives = _get_lambda_derivatives(Xl_train_ordered, y_train_ordered, Xl_validate_ordered, y_validate_ordered, beta, thetas, train_indices, validate_indices, difference_matrix, acc_regularization)

    method_step_size = STEP_SIZE
    potential_cost = current_cost * 10
    while potential_cost > current_cost:
        if np.any(np.isnan(lambda_derivatives)):
            print "first time: some value in df_dlambda is nan"
            break

        potential_curr_regularization = _get_updated_lambdas(acc_regularization, method_step_size, lambda_derivatives, boundary_factor=BOUNDARY_FACTOR)
        potential_beta, potential_thetas, potential_cost = _get_beta_theta_cost(problem_wrapper, Xl_validate_ordered, y_validate_ordered, potential_curr_regularization)

        print "first time: new proposal: potential_curr_regularization", potential_curr_regularization, "method_step_size", method_step_size, "potential_cost", potential_cost
        if potential_cost > current_cost:
            if potential_cost > 2 * current_cost:
                method_step_size *= SHRINK_SHRINK_FACTOR * 0.01
            else:
                method_step_size *= SHRINK_SHRINK_FACTOR
                if method_step_size < METHOD_STEP_SIZE_MIN:
                    print "METHOD STEP SIZE TOO SMALL"
                    return beta, thetas, cost_path, acc_regularization

    print "I FOUND A GOOD STEP SIZE", method_step_size

    # Perform Nesterov with adaptive restarts
    i_max = 3
    while i_max > 2:
        print "restart! with i_max", i_max
        for i in range(2, NUMBER_OF_ITERATIONS + 1):
            i_max = i
            if i != 2:
                try:
                    lambda_derivatives = _get_lambda_derivatives(Xl_train_ordered, y_train_ordered, Xl_validate_ordered, y_validate_ordered, beta, thetas, train_indices, validate_indices, difference_matrix, acc_regularization)
                except np.linalg.LinAlgError as e:
                    print "linalg error. returning early"
                    print e
                    break

            if np.any(np.isnan(lambda_derivatives)):
                print "some value in df_dlambda is nan"
                break

            curr_regularization = _get_updated_lambdas(acc_regularization, method_step_size, lambda_derivatives)
            acc_regularization = _get_accelerated_lambdas(curr_regularization, prev_regularization, i)
            print "new proposal: acc_regularization", acc_regularization
            prev_regularization = curr_regularization

            potential_beta, potential_thetas, potential_cost = _get_beta_theta_cost(problem_wrapper, Xl_validate_ordered, y_validate_ordered, acc_regularization)

            if potential_cost > cost_path[-1]:
                print "COST IS INCREASING. break."
                break
            else:
                current_cost = potential_cost
                beta = potential_beta
                thetas = potential_thetas
                cost_path.append(current_cost)

                print method_label, "iter:", i - 2, "current_cost:", current_cost, "lambdas:", acc_regularization
                if cost_path[-2] - cost_path[-1] < DECREASING_ENOUGH_THRESHOLD:
                    print "progress too slow", cost_path[-2] - cost_path[-1]
                    break

    print method_label, "best cost", current_cost, "curr lambdas:", acc_regularization, "total iters:", i - 2
    print method_label, "first cost", cost_path[0], "first_regularization", first_regularization

    return beta, thetas, cost_path, acc_regularization


def _get_updated_lambdas(lambdas, method_step_size, lambda_derivatives, boundary_factor=None):
    new_step_size = method_step_size
    if boundary_factor is not None:
        potential_lambdas = lambdas - method_step_size * lambda_derivatives

        for idx in range(0, lambdas.size):
            if lambdas[idx] > LAMBDA_MIN and potential_lambdas[idx] < (1 - BOUNDARY_FACTOR) * lambdas[idx]:
                smaller_step_size = BOUNDARY_FACTOR * lambdas[idx] / lambda_derivatives[idx]
                new_step_size = min(new_step_size, smaller_step_size)

    return np.maximum(lambdas - new_step_size * lambda_derivatives, LAMBDA_MIN)

def _get_lambda_derivatives(Xl_train, y_train, Xl_validate, y_validate, beta, thetas, train_indices, validate_indices, difference_matrix, lambdas):
    # beta_nonzero_indices = get_nonzero_indices(beta)
    # Xl_train_mini = Xl_train[:, beta_nonzero_indices]
    # Xl_validate_mini = Xl_validate[:, beta_nonzero_indices]
    # beta_mini = beta[beta_nonzero_indices]
    #
    # if beta_mini.size == 0:
    #     return np.array([0] * len(lambdas))
    #
    #
    # return _get_lambda_derivatives_mini(Xl_train_mini, y_train, Xl_validate_mini, y_validate, beta_mini, thetas, train_indices, validate_indices, difference_matrix, lambdas)
    return _get_lambda_derivatives_mini(Xl_train, y_train, Xl_validate, y_validate, beta, thetas, train_indices, validate_indices, difference_matrix, lambdas)

def _get_lambda_derivatives_mini(Xl_train, y_train, Xl_validate, y_validate, beta, thetas, train_indices, validate_indices, difference_matrix, lambdas):
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

    inv_matrix12 = sp.linalg.pinvh(MM + lambdas[1] * DD)
    inv_matrix12_X = inv_matrix12 * M.T * Xl_train
    matrix12_to_inv = XX - Xl_train.T * M * inv_matrix12_X
    dbeta_dlambdas[0], _, _, _ = np.linalg.lstsq(matrix12_to_inv, -1 * _get_zero_or_sign_vector(beta))
    dtheta_dlambdas[0] = -1 * inv_matrix12_X * dbeta_dlambdas[0]

    inv_matrix34_X = sp.linalg.pinvh(XX) * Xl_train.T * M
    matrix34_to_inv = MM + lambdas[1] * DD - M.T * Xl_train * inv_matrix34_X

    dtheta_dlambdas[1], _, _, _ = np.linalg.lstsq(matrix34_to_inv, -1.0 * difference_matrix.T * difference_matrix * thetas)
    dbeta_dlambdas[1] = -1 * inv_matrix34_X * dtheta_dlambdas[1]
    err_vector = y_validate - Xl_validate * beta - thetas[validate_indices]

    df_dlambdas = [
        -1 * ((Xl_validate * dbeta_dlambdas[i] + dtheta_dlambdas[i][validate_indices]).T * err_vector)[0,0]
        for i in range(0, len(lambdas))
    ]

    print "df_dlambdas", df_dlambdas

    return np.array(df_dlambdas)
