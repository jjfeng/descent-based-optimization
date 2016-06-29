import scipy as sp
import numpy as np
from common import *
from convexopt_solvers import GroupedLassoProblemWrapperSimple

NUMBER_OF_ITERATIONS = 40
STEP_SIZE = 1
BOUNDARY_FACTOR = 0.5
DECREASING_ENOUGH_THRESHOLD = 1e-4 * 5
SHRINK_SHRINK = 0.1
SHRINK_INIT = 1
MIN_SHRINK = 1e-6
MIN_METHOD_STEP = 1e-10
MIN_LAMBDA = 1e-6
BACKTRACK_ALPHA = 0.001

def run(X_train, y_train, X_validate, y_validate, group_feature_sizes, initial_lambda=1):
    method_step_size = STEP_SIZE
    shrink_factor = SHRINK_INIT

    curr_regularizations = np.array([initial_lambda] * 2)
    problem_wrapper = GroupedLassoProblemWrapperSimple(X_train, y_train, group_feature_sizes)
    betas = problem_wrapper.solve(curr_regularizations)
    best_beta = betas
    best_cost = testerror_grouped(X_validate, y_validate, betas)
    cost_path = [best_cost]

    for i in range(0, NUMBER_OF_ITERATIONS):
        print "ITER", i, "current lambdas", curr_regularizations

        lambda_derivatives = _get_lambda_derivatives(X_train, y_train, X_validate, y_validate, betas, curr_regularizations)
        if np.any(np.isnan(lambda_derivatives)):
            break

        # do the gradient descent
        pot_lambdas = _get_updated_lambdas(curr_regularizations, shrink_factor * method_step_size, lambda_derivatives)

        # get corresponding beta
        pot_betas = problem_wrapper.solve(pot_lambdas)
        if pot_betas is None:
            pot_cost = np.inf
        else:
            pot_cost = testerror_grouped(X_validate, y_validate, pot_betas)

        # Do bactracking line descent if needed
        backtrack_check = best_cost - BACKTRACK_ALPHA * shrink_factor * method_step_size * np.linalg.norm(lambda_derivatives)**2
        backtrack_check = best_cost if backtrack_check < 0 else backtrack_check
        while pot_cost > backtrack_check and shrink_factor > MIN_SHRINK:
            shrink_factor *= SHRINK_SHRINK
            pot_lambdas = _get_updated_lambdas(curr_regularizations, shrink_factor * method_step_size, lambda_derivatives)
            pot_betas = problem_wrapper.solve(pot_lambdas)
            if pot_betas is None:
                pot_cost = np.inf
            else:
                pot_cost = testerror_grouped(X_validate, y_validate, pot_betas)

            backtrack_check = best_cost - BACKTRACK_ALPHA * shrink_factor * method_step_size * np.linalg.norm(lambda_derivatives)**2
            backtrack_check = best_cost if backtrack_check < 0 else backtrack_check

        is_decreasing_significantly = best_cost - pot_cost > DECREASING_ENOUGH_THRESHOLD

        betas = pot_betas
        curr_regularizations = pot_lambdas
        if pot_cost < best_cost:
            best_cost = pot_cost
            best_beta = betas

        if not is_decreasing_significantly:
            break

        if shrink_factor <= MIN_SHRINK:
            break

        cost_path.append(pot_cost)

    return best_beta, cost_path

def _get_updated_lambdas(lambdas, method_step_size, lambda_derivatives, use_boundary=False):
    new_step_size = method_step_size
    if use_boundary:
        potential_lambdas = lambdas - method_step_size * lambda_derivatives

        for idx in range(0, lambdas.size):
            if lambdas[idx] > MIN_LAMBDA and potential_lambdas[idx] < (1 - BOUNDARY_FACTOR) * lambdas[idx]:
                smaller_step_size = BOUNDARY_FACTOR * lambdas[idx] / lambda_derivatives[idx]
                new_step_size = min(new_step_size, smaller_step_size)

    return np.maximum(lambdas - new_step_size * lambda_derivatives, MIN_LAMBDA)

def _get_lambda_derivatives(X_train, y_train, X_validate, y_validate, betas, curr_regularizations):
    # minify the data for computational speed
    beta_minis = []
    beta_nonzeros = []
    for beta in betas:
        beta_nonzero = get_nonzero_indices(beta)
        beta_nonzeros.append(beta_nonzero)
        beta_minis.append(beta[beta_nonzero])

    complete_beta_nonzero = np.concatenate(beta_nonzeros)
    X_train_mini = X_train[:, complete_beta_nonzero]
    X_validate_mini = X_validate[:, complete_beta_nonzero]

    if X_train_mini.size == 0:
        return np.array([0] * len(curr_regularizations))

    return _get_lambda_derivatives_mini(X_train_mini, y_train, X_validate_mini, y_validate, beta_minis, curr_regularizations)

def _get_lambda_derivatives_mini(X_train, y_train, X_validate, y_validate, betas, curr_regularizations):
    def _get_block_diag_component(beta):
        if beta.size == 0:
            return np.matrix(np.zeros((0,0))).T

        repeat_hstacked_beta = np.tile(beta, (1, beta.size)).T
        block_diag_component = get_norm2(beta, power=-3) * np.diagflat(beta) * repeat_hstacked_beta
        return block_diag_component

    def _get_diagmatrix_component(beta):
        if beta.size == 0:
            return np.matrix(np.zeros((0,0))).T
        return get_norm2(beta, power=-1) * np.identity(beta.size)

    def _get_dbeta_dlambda1(matrix_to_invert):
        normed_beta = np.concatenate([beta / get_norm2(beta) if beta.size > 0 else np.zeros(beta.shape) for beta in betas])
        dbeta_dlambda1, _, _, _ = np.linalg.lstsq(matrix_to_invert, -1 * normed_beta)
        return dbeta_dlambda1

    complete_beta = np.concatenate(betas)
    lambda1 = curr_regularizations[0]

    block_diag_components = [_get_block_diag_component(beta) for beta in betas]
    diagonal_components = [_get_diagmatrix_component(beta) for beta in betas]
    dgrouplasso_dlambda = -1 * lambda1 * sp.linalg.block_diag(*block_diag_components) + lambda1 * sp.linalg.block_diag(*diagonal_components)

    matrix_to_invert = 1.0 / y_train.size * X_train.T * X_train + dgrouplasso_dlambda

    normed_beta = np.concatenate([beta / get_norm2(beta) if beta.size > 0 else np.zeros(beta.shape) for beta in betas])

    inverted_m = sp.linalg.pinvh(matrix_to_invert)
    dbeta_dlambda1 = inverted_m * -1 * normed_beta
    dbeta_dlambda2 = inverted_m * -1 * np.sign(complete_beta)

    err_vector = y_validate - X_validate * complete_beta
    df_dlambda1 = -1.0 / y_validate.size * (X_validate * dbeta_dlambda1).T * err_vector
    df_dlambda2 = -1.0 / y_validate.size * (X_validate * dbeta_dlambda2).T * err_vector

    return np.array([df_dlambda1[0,0], df_dlambda2[0,0]])
