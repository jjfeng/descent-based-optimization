import time
import matplotlib.pyplot as plt
from common import *
import data_generation
import hillclimb_elasticnet_lambda12 as hc
import gridsearch_elasticnet_lambda12
from method_results import MethodResult
from method_results import MethodResults

NUM_RUNS = 30

SIGNAL_NOISE_RATIO = 2
NUM_FEATURES = 250
NUM_NONZERO_FEATURES = 15
TRAIN_SIZE = 80
COARSE_LAMBDA_GRID = [1e-2, 1e1]
NUM_RANDOM_LAMBDAS = 3

seed = int(np.random.rand() * 1e5)
np.random.seed(seed)
print "SEED", seed

def _hillclimb_coarse_grid_search(optimization_func, *args, **kwargs):
    start_time = time.time()
    best_cost = 1e10
    best_beta = []
    best_start_lambdas = []
    for init_lambda1 in COARSE_LAMBDA_GRID:
        kwargs["initial_lambda1"] = init_lambda1
        kwargs["initial_lambda2"] = init_lambda1
        beta_guess, cost_path = optimization_func(*args, **kwargs)
        validation_cost = testerror(X_validate, y_validate, beta_guess)
        if best_cost > validation_cost:
            best_start_lambdas = [kwargs["initial_lambda1"], kwargs["initial_lambda2"]]
            best_cost = validation_cost
            best_beta = beta_guess
            best_cost_path = cost_path

    end_time = time.time()
    print "HC: BEST best_cost", best_cost, "best_start_lambdas", best_start_lambdas
    return beta_guess, cost_path, end_time - start_time

hc_results = MethodResults(HC_LAMBDA12_LABEL)
gs_results = MethodResults(GS_LAMBDA12_LABEL)

for i in range(0, NUM_RUNS):
    beta_real, X_train, y_train, X_validate, y_validate, X_test, y_test = data_generation.correlated(
        TRAIN_SIZE, NUM_FEATURES, NUM_NONZERO_FEATURES, signal_noise_ratio=SIGNAL_NOISE_RATIO)

    def _create_method_result(beta_guess, runtime):
        # helper method for aggregating results
        test_err = testerror(X_test, y_test, beta_guess)
        validation_err = testerror(X_validate, y_validate, beta_guess)
        beta_err = betaerror(beta_real, beta_guess)
        return MethodResult(test_err=test_err, beta_err=beta_err, validation_err=validation_err, runtime=runtime)

    hc_beta_guess, hc_cost_path, runtime = _hillclimb_coarse_grid_search(hc.run, X_train, y_train, X_validate, y_validate)
    hc_results.append(_create_method_result(hc_beta_guess, runtime))

    start = time.time()
    gs_beta_guess = gridsearch_elasticnet_lambda12.run(X_train, y_train, X_validate, y_validate)
    runtime = time.time() - start
    gs_method_result = _create_method_result(gs_beta_guess, runtime)
    gs_results.append(gs_method_result)

    print "===========RUN %d ============" % i
    hc_results.print_results()
    gs_results.print_results()
