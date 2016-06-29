import sys
import scipy as sp
import numpy as np
from common import *
from realdata_colitis_models import AllKFoldsData
from convexopt_solvers import GroupedLassoClassifyProblemWrapperFullCV

NUMBER_OF_ITERATIONS = 15
STEP_SIZE = 1
DECREASING_ENOUGH_THRESHOLD = 0.05
SHRINK_SHRINK = 0.01
SHRINK_INIT = 1
MIN_SHRINK = 1e-6
MIN_LAMBDA = 1e-16

def run_for_lambdas(X_groups_train_validate, y_train_validate, feature_group_sizes, kfolds, init_lambdas=[]):
    # Allow to run gradient descent with multiple start points
    
    X_train_validate = np.hstack(X_groups_train_validate)

    full_problem = GroupedLassoClassifyProblemWrapperFullCV(X_train_validate, y_train_validate, feature_group_sizes)
    all_kfolds_data = AllKFoldsData(X_train_validate, y_train_validate, feature_group_sizes, kfolds, GroupedLassoClassifyProblemWrapperFullCV)

    # Add one for lambda2
    ones = np.ones(len(feature_group_sizes) + 1)

    best_lambdas = []
    best_cost = np.inf
    hc_cost_path = []
    best_start_lambda = 0
    for init_lambda in init_lambdas:
        perturbed_inits = ones * init_lambda + np.random.randn(len(feature_group_sizes) + 1) * init_lambda / 5.0
        lambda_vals, validate_cost, cost_path = run(all_kfolds_data, perturbed_inits)
        if validate_cost < best_cost:
            best_cost = validate_cost
            best_lambdas = lambda_vals
            hc_cost_path = cost_path
            best_start_lambda = init_lambda

        # If this lambda fails terribly, then try a different one
        if len(cost_path) > 3:
            break

    beta = full_problem.solve(best_lambdas)
    return beta, best_cost, cost_path

def run(all_kfolds_data, init_lambdas):
    """
    Returns best_regularizations and best_cost
    """
    method_step_size = STEP_SIZE

    best_regularizations = init_lambdas

    betas, best_cost = all_kfolds_data.solve(best_regularizations)
    all_kfolds_data.set_betas(betas)

    # track progression
    cost_path = [best_cost]

    shrink_factor = SHRINK_INIT
    for i in range(0, NUMBER_OF_ITERATIONS):
        print "ITER", i, "current lambdas", best_regularizations

        lambda_derivatives = all_kfolds_data.get_lambda_derivatives(best_regularizations)

        if np.linalg.norm(lambda_derivatives) < 1e-16:
            break

        # do the gradient descent!
        pot_lambdas = _get_updated_lambdas(best_regularizations, shrink_factor * method_step_size, lambda_derivatives)

        # get corresponding beta
        betas, pot_cost = all_kfolds_data.solve(pot_lambdas)

        while pot_cost >= best_cost and shrink_factor > MIN_SHRINK:
            shrink_factor *= SHRINK_SHRINK
            pot_lambdas = _get_updated_lambdas(best_regularizations, shrink_factor * method_step_size, lambda_derivatives)
            betas, pot_cost = all_kfolds_data.solve(pot_lambdas)

        is_decreasing_significantly = best_cost - pot_cost > DECREASING_ENOUGH_THRESHOLD

        if pot_cost < best_cost:
            best_cost = pot_cost
            best_regularizations = pot_lambdas
            all_kfolds_data.set_betas(betas)

        if not is_decreasing_significantly:
            break

        if shrink_factor <= MIN_SHRINK:
            break

        cost_path.append(pot_cost)

    return best_regularizations, best_cost, cost_path


def _get_updated_lambdas(lambdas, method_step_size, lambda_derivatives):
    return np.maximum(lambdas - method_step_size * lambda_derivatives, MIN_LAMBDA)
