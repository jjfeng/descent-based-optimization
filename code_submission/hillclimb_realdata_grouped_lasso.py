import sys
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import numpy as np
from common import *
from convexopt_solvers import GroupedLassoClassifyProblemWrapper

NUMBER_OF_ITERATIONS = 40
STEP_SIZE = 1
DECREASING_ENOUGH_THRESHOLD = 0.008
SHRINK_SHRINK = 0.01
MIN_SHRINK = 1e-8
MIN_LAMBDA = 1e-16

def run_for_lambdas(X_groups_train, y_train, X_groups_validate, y_validate, init_lambdas=[]):
    problem_wrapper = GroupedLassoClassifyProblemWrapper(X_groups_train, y_train)

    X_validate = np.hstack(X_groups_validate)
    X_train = np.hstack(X_groups_train)

    hc_betas = []
    hc_validate_cost = 1e10
    hc_cost_path = []
    ones = np.ones(len(X_groups_train) + 1)
    best_start_lambda = 0
    hc_validate_rate = 0
    for init_lambda in init_lambdas:
        print HC_GROUPED_LASSO_LABEL, "try lambda", init_lambda
        betas, cost_path = run(problem_wrapper, X_train, y_train, X_validate, y_validate, ones * init_lambda)
        validate_cost, classification_rate = testerror_logistic_grouped(X_validate, y_validate, betas)
        if validate_cost < hc_validate_cost:
            hc_validate_cost = validate_cost
            hc_validate_rate = classification_rate
            hc_betas = betas
            hc_cost_path = cost_path
            best_start_lambda = init_lambda

    print HC_GROUPED_LASSO_LABEL, "best start lambda", best_start_lambda, "hc_validate_cost", hc_validate_cost, "hc_validate_rate", hc_validate_rate

    return hc_betas, cost_path

def run(problem_wrapper, X_train, y_train, X_validate, y_validate, init_lambdas):
    method_step_size = STEP_SIZE

    curr_regularizations = init_lambdas

    betas = problem_wrapper.solve(curr_regularizations)
    best_beta = betas
    best_cost, _ = testerror_logistic_grouped(X_validate, y_validate, betas)
    print "first cost", best_cost

    # track progression
    cost_path = [best_cost]

    shrink_factor = 1
    for i in range(0, NUMBER_OF_ITERATIONS):
        lambda_derivatives = _get_lambda_derivatives(X_train, y_train, X_validate, y_validate, betas, curr_regularizations)
        print "lambda_derivatives norm", np.linalg.norm(lambda_derivatives)

        if np.linalg.norm(lambda_derivatives) < 1e-16:
            print HC_GROUPED_LASSO_LABEL, "lambda derivatives are zero!"
            break

        sys.stdout.flush()

        # do the gradient descent!
        pot_lambdas = _get_updated_lambdas(curr_regularizations, shrink_factor * method_step_size, lambda_derivatives)

        # get corresponding beta
        pot_betas = problem_wrapper.solve(pot_lambdas)

        try:
            pot_cost, _ = testerror_logistic_grouped(X_validate, y_validate, pot_betas)
            print HC_GROUPED_LASSO_LABEL, "pot_cost", pot_cost
            sys.stdout.flush()
        except ValueError as e:
            print "value error", e
            pot_cost = 1e10

        while pot_cost >= best_cost and shrink_factor > MIN_SHRINK:
            shrink_factor *= SHRINK_SHRINK
            print HC_GROUPED_LASSO_LABEL, "shrink!", shrink_factor
            pot_lambdas = _get_updated_lambdas(curr_regularizations, shrink_factor * method_step_size, lambda_derivatives)
            pot_betas = problem_wrapper.solve(pot_lambdas)
            try:
                pot_cost, _ = testerror_logistic_grouped(X_validate, y_validate, pot_betas)
            except ValueError as e:
                print "value error", e
                pot_cost = 1e10
            print HC_GROUPED_LASSO_LABEL, "pot_cost", pot_cost

        is_decreasing_signficantly = best_cost - pot_cost > DECREASING_ENOUGH_THRESHOLD

        betas = pot_betas
        curr_regularizations = pot_lambdas
        if pot_cost < best_cost:
            best_cost = pot_cost
            best_beta = betas

        if not is_decreasing_signficantly:
            print HC_GROUPED_LASSO_LABEL, "is_decreasing_signficantly NO!"
            break

        if shrink_factor <= MIN_SHRINK:
            print HC_GROUPED_LASSO_LABEL, "shrink factor too small"
            break

        # track progression
        cost_path.append(pot_cost)

        print HC_GROUPED_LASSO_LABEL, "iter", i, "best cost", best_cost, "lambdas:", curr_regularizations
        sys.stdout.flush()

    print HC_GROUPED_LASSO_LABEL, "best cost", best_cost, "lambdas:", curr_regularizations, "total_iters", i + 1
    return best_beta, cost_path


def _get_updated_lambdas(lambdas, method_step_size, lambda_derivatives):
    return np.maximum(lambdas - method_step_size * lambda_derivatives, MIN_LAMBDA)


def _get_lambda_derivatives(X_train, y_train, X_validate, y_validate, betas, curr_regularizations):
    # first minify the data
    beta_minis = []
    beta_nonzeros = []
    for beta in betas:
        # print "beta", beta
        beta_nonzero = get_nonzero_indices(beta, threshold=1e-5)
        beta_nonzeros.append(beta_nonzero)
        beta_minis.append(beta[beta_nonzero])

    complete_beta_nonzero = np.concatenate(beta_nonzeros)
    X_train_mini = X_train[:, complete_beta_nonzero]
    X_validate_mini = X_validate[:, complete_beta_nonzero]

    if X_train_mini.size == 0:
        return np.array([0] * len(curr_regularizations))

    return _get_lambda_derivatives_mini(X_train_mini, y_train, X_validate_mini, y_validate, beta_minis, curr_regularizations)

def _get_lambda_derivatives_mini(X_train, y_train, X_validate, y_validate, betas, curr_regularizations):
    lambda1s = curr_regularizations[0:-1]

    def _get_block_diag_component(idx):
        beta = betas[idx]
        if beta.size == 0:
            return np.matrix(np.zeros((0,0))).T

        repeat_hstacked_beta = np.tile(beta, (1, beta.size)).T
        block_diag_component = -1 * lambda1s[idx] / get_norm2(beta, power=3) * np.diagflat(beta) * repeat_hstacked_beta
        return block_diag_component

    def _get_diagmatrix_component(idx):
        beta = betas[idx]
        if beta.size == 0:
            return np.matrix(np.zeros((0,0))).T
        return lambda1s[idx] / get_norm2(beta) * np.identity(beta.size)

    def _get_dbeta_dlambda1(beta, inverted_matrix, num_features_before):
    # def _get_dbeta_dlambda1(beta, matrix_to_invert, num_features_before):
        if beta.size == 0:
            return np.matrix(np.zeros((inverted_matrix.shape[0], 1)))
        else:
            normed_beta = beta / get_norm2(beta)
            zero_normed_beta = np.concatenate([
                np.matrix(np.zeros(num_features_before)).T,
                normed_beta,
                np.matrix(np.zeros(total_features - normed_beta.size - num_features_before)).T
            ])

            dbeta_dlambda1 = -1 * inverted_matrix * zero_normed_beta
            return dbeta_dlambda1

    num_feature_groups = len(betas)
    total_features = X_train.shape[1]
    complete_beta = np.matrix(np.concatenate(betas))

    exp_Xb = np.matrix(np.exp(X_train * complete_beta))
    diag_expXb_components = np.diagflat(np.multiply(np.power(1 + exp_Xb, -2), exp_Xb))

    block_diag_components = [_get_block_diag_component(idx) for idx in range(0, num_feature_groups)]
    diagonal_components = [_get_diagmatrix_component(idx) for idx in range(0, num_feature_groups)]
    dgrouplasso_dlambda = sp.linalg.block_diag(*block_diag_components) + sp.linalg.block_diag(*diagonal_components)

    print "X_train", X_train.shape
    matrix_to_invert = X_train.T * diag_expXb_components * X_train + dgrouplasso_dlambda
    print "matrix_to_invert min", np.amin(np.abs(matrix_to_invert))
    # matrix_to_invert[np.less(np.abs(matrix_to_invert), 1e-10)] = 0
    print "matrix_to_invert max", np.amax(np.abs(matrix_to_invert))
    print "matrix_to_invert sum", np.sum(matrix_to_invert)
    print "matrix_to_invert shape", matrix_to_invert.shape
    print "matrix_to_invert", matrix_to_invert

    inverted_matrix = sp.linalg.pinvh(matrix_to_invert)
    print "matrix_to_invert inverted!"

    dbeta_dlambda1s = np.matrix(np.zeros((0,0))).T
    num_features_before = 0

    for beta in betas:
        dbeta_dlambda1 = _get_dbeta_dlambda1(beta, inverted_matrix, num_features_before)
        # dbeta_dlambda1 = _get_dbeta_dlambda1(beta, matrix_to_invert, num_features_before)
        num_features_before += beta.size

        if dbeta_dlambda1s.size == 0:  # not initialized yet
            dbeta_dlambda1s = dbeta_dlambda1
        else:
            dbeta_dlambda1s = np.hstack([dbeta_dlambda1s, dbeta_dlambda1])

    dbeta_dlambda2 = inverted_matrix * -1 * np.sign(complete_beta)
    # dbeta_dlambda2, _, _, _ = np.linalg.lstsq(matrix_to_invert, -1 * np.sign(complete_beta))

    expXvBeta = np.exp(X_validate * complete_beta)
    dloss_dbeta = X_validate.T * (-1 * y_validate + 1 - np.power(1 + expXvBeta, -1))
    df_dlambda1s = dloss_dbeta.T * dbeta_dlambda1s
    df_dlambda1s = np.reshape(np.array(df_dlambda1s), df_dlambda1s.size)
    df_dlambda2 = dloss_dbeta.T * dbeta_dlambda2
    return np.concatenate((df_dlambda1s, [df_dlambda2[0,0]]))
