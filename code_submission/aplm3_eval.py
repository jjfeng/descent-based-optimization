import sys
import getopt
import time
import cvxpy
from common import *
from method_results import MethodResult
from method_results import MethodResults
from data_generation import smooth_plus_linear
import hillclimb_smooth_add_linear as hc
import gridsearch_smooth_add_linear as gs

NUM_RUNS = 30

NUM_TRAIN = 100
NUM_FEATURES = 20
NUM_NONZERO_FEATURES = 6

SIGNAL_TO_NOISE = 2
LINEAR_TO_SMOOTH_RATIO = 2

COARSE_LAMBDAS = [1e-2, 1e-1, 1, 10]

def _hillclimb_coarse_grid_search(optimization_func, *args, **kwargs):
    # Allow multiple start points for grid search
    start_time = time.time()
    best_cost = np.inf
    best_beta = []
    best_thetas = []
    best_cost_path = []
    best_regularization = []
    best_start_lambda = []
    for l1 in COARSE_LAMBDAS:
        kwargs["initial_lambdas"] = [l1,l1,l1]

        try:
            beta, thetas, cost_path, curr_regularization = optimization_func(*args, **kwargs)
        except cvxpy.error.SolverError as e:
            # cvxpy was not able to find a soln for this gradient descent initialization
            continue

        if beta is not None and thetas is not None and best_cost > cost_path[-1]:
            best_cost = cost_path[-1]
            best_beta = beta
            best_thetas = thetas
            best_cost_path = cost_path
            best_start_lambda = [l1]
            best_regularization = curr_regularization

    end_time = time.time()
    return best_beta, best_thetas, best_cost_path, end_time - start_time

def _get_ordered_Xl_y_data(Xl, Xs, y):
    indices = np.argsort(np.reshape(np.array(Xs), Xs.size), axis=0)
    return Xl[indices, :], y[indices]

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"d:")
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-d':
            data_type = int(arg)

    assert(data_type in [0, 1, 2])

    hc_results = MethodResults("Hillclimb")
    gs_results = MethodResults("Gridsearch")

    for i in range(0, NUM_RUNS):
        # Generate data
        beta_real, Xl_train, Xs_train, y_train, Xl_validate, Xs_validate, y_validate, Xl_test, Xs_test, y_test, y_smooth_train, y_smooth_validate, y_smooth_test = smooth_plus_linear(
            NUM_TRAIN,
            NUM_FEATURES,
            NUM_NONZERO_FEATURES,
            data_type=data_type,
            linear_to_smooth_ratio=LINEAR_TO_SMOOTH_RATIO,
            desired_signal_noise_ratio=SIGNAL_TO_NOISE)

        tot_size = Xs_train.size + Xs_validate.size + Xs_test.size
        Xs_combined = np.reshape(np.array(np.vstack((Xs_train, Xs_validate, Xs_test))), tot_size)
        order_indices = np.argsort(Xs_combined, axis=0)

        test_indices = np.greater_equal(order_indices, Xs_train.size + Xs_validate.size)
        validate_indices = np.logical_and(np.greater_equal(order_indices, Xs_train.size), np.logical_not(test_indices))
        train_validate_indices = np.logical_not(test_indices)

        Xs_ordered = Xs_combined[order_indices]
        true_y_smooth = np.matrix(np.vstack((y_smooth_train, y_smooth_validate, y_smooth_test))[order_indices])

        Xl_test_ordered, y_test_ordered = _get_ordered_Xl_y_data(Xl_test, Xs_test, y_test)
        Xl_validate_ordered, y_validate_ordered = _get_ordered_Xl_y_data(Xl_validate, Xs_validate, y_validate)

        def _create_method_result(best_beta, best_thetas, runtime):
            # Helper function for aggregating the results
            beta_err = betaerror(beta_real, best_beta)
            theta_err = betaerror(true_y_smooth, best_thetas)
            test_err = testerror_smooth_and_linear(Xl_test_ordered, y_test_ordered, best_beta, best_thetas[test_indices])
            validation_err = testerror_smooth_and_linear(Xl_validate_ordered, y_validate_ordered, best_beta, best_thetas[validate_indices])
            return MethodResult(test_err=test_err, beta_err=beta_err, theta_err=theta_err, validation_err=validation_err, runtime=runtime)

        # Run gradient descent
        hc_beta, hc_thetas, hc_cost_path, runtime = _hillclimb_coarse_grid_search(hc.run, Xl_train, Xs_train, y_train, Xl_validate, Xs_validate, y_validate, Xs_test)
        assert(hc_beta is not None and hc_thetas is not None)
        hc_results.append(_create_method_result(hc_beta, hc_thetas, runtime))

        # Run grid search
        start_time = time.time()
        gs_beta, gs_thetas, gs_best_cost = gs.run(Xl_train, Xs_train, y_train, Xl_validate, Xs_validate, y_validate, Xs_test)
        gs_results.append_runtime(time.time() - start_time)
        gs_results.append(_create_method_result(gs_beta, gs_thetas, runtime))

        print "===========RUN %d ============" % i
        gs_results.print_results()
        hc_results.print_results()

if __name__ == "__main__":
    main(sys.argv[1:])
