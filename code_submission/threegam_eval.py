import time
import sys
import numpy as np

from convexopt_solvers import GenAddModelProblemWrapper
from data_generation import multi_smooth_features

from gen_add_model_hillclimb import GenAddModelHillclimb
from method_results import MethodResults
from method_results import MethodResult
import gen_add_model_gridsearch as gs

from common import *

FEATURE_RANGE = [-5.0, 5.0]
NUM_RUNS = 30

NUM_FUNCS = 3
TRAIN_SIZE = 180
SNR = 2
VALIDATE_RATIO = 3
NUM_TEST = 60
TEST_HC_LAMBDAS = [1]

NUM_GS_LAMBDAS = 10
MAX_LAMBDA = 50

# The three true functions that sum together into y (modulo an error term)
def identity_fcn(x):
    return x.reshape(x.size, 1)

def big_sin(x):
    return identity_fcn(9 * np.sin(x*2))

def big_cos_sin(x):
    return identity_fcn(6 * (np.cos(x * 1.25) + np.sin(x/2 + 0.5)))

def _hillclimb_coarse_grid_search(hc, smooth_fcn_list):
    # Do gradient descent, possibly from multiple starting points
    start_time = time.time()
    best_cost = np.inf
    best_thetas = []
    best_cost_path = []
    best_regularization = []
    best_start_lambda = []
    for lam in TEST_HC_LAMBDAS:
        init_lambdas = np.array([lam for i in range(len(smooth_fcn_list))])
        thetas, cost_path, curr_regularization = hc.run(init_lambdas)

        if thetas is not None and best_cost > cost_path[-1]:
            best_cost = cost_path[-1]
            best_thetas = thetas
            best_cost_path = cost_path
            best_start_lambda = lam
            best_regularization = curr_regularization

    end_time = time.time()
    return best_thetas, best_cost_path, end_time - start_time

def main():
    smooth_fcn_list = [big_sin, identity_fcn, big_cos_sin]

    hc_results = MethodResults("Hillclimb")
    gs_results = MethodResults("Gridsearch")

    for i in range(0, NUM_RUNS):
        # Generate dataset
        X_train, y_train, X_validate, y_validate, X_test, y_test = multi_smooth_features(
            TRAIN_SIZE,
            smooth_fcn_list,
            desired_snr=SNR,
            feat_range=[f * TRAIN_SIZE/60 for f in FEATURE_RANGE],
            train_to_validate_ratio=VALIDATE_RATIO,
            test_size=NUM_TEST
        )
        X_full, train_idx, validate_idx, test_idx = GenAddModelHillclimb.stack((X_train, X_validate, X_test))

        def _create_method_result(best_thetas, runtime):
            # Helper function for aggregating the results
            test_err = testerror_multi_smooth(y_test, test_idx, best_thetas)
            validate_err = testerror_multi_smooth(y_validate, validate_idx, best_thetas)
            return MethodResult(test_err=test_err, validation_err=validate_err, runtime=runtime)

        # Run gradient descent
        hillclimb_prob = GenAddModelHillclimb(X_train, y_train, X_validate, y_validate, X_test)
        hc_thetas, hc_cost_path, runtime = _hillclimb_coarse_grid_search(hillclimb_prob, smooth_fcn_list)
        hc_results.append(_create_method_result(hc_thetas, runtime))

        # Run grid search
        start_time = time.time()
        gs_thetas, best_lambdas = gs.run(
            y_train,
            y_validate,
            X_full,
            train_idx,
            validate_idx,
            num_lambdas=NUM_GS_LAMBDAS,
            max_lambda=MAX_LAMBDA
        )
        gs_runtime = time.time() - start_time
        gs_results.append(_create_method_result(gs_thetas, gs_runtime))

        print "===========RUN %d ============" % i
        hc_results.print_results()
        gs_results.print_results()

if __name__ == "__main__":
    main()
