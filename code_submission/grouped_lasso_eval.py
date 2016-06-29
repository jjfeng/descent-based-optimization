import sys
import getopt
import time
from method_results import MethodResults
from method_results import MethodResult
from data_generation import sparse_groups
import hillclimb_grouped_lasso as hc
import hillclimb_pooled_grouped_lasso as hc_pooled
import gridsearch_grouped_lasso
from common import *

NUM_RUNS = 30

TRUE_NUM_GROUPS = 3
ZERO_THRESHOLD = 0.5 * 1e-4

def main(argv):
    try:
        # -d: decide what the true number of groups are and the number of groups hypothesized
        # -p: if included, this runs sparse group lasso. if excluded, this runs unpooled sparse group lasso
        opts, args = getopt.getopt(argv,"d:p")
    except getopt.GetoptError:
        sys.exit(2)

    RUN_HC_POOLED = False
    for opt, arg in opts:
        if opt == '-d':
            data_type = int(arg)
            if data_type == 1:
                TRAIN_SIZE = 60
                TOTAL_FEATURES = 300
                NUM_GROUPS = 30
            elif data_type == 2:
                TRAIN_SIZE = 90
                TOTAL_FEATURES = 900
                NUM_GROUPS = 60
            elif data_type == 3:
                TRAIN_SIZE = 90
                TOTAL_FEATURES = 1200
                NUM_GROUPS = 100
        elif opt == '-p':
            RUN_HC_POOLED = True

    TRUE_GROUP_FEATURE_SIZES = [TOTAL_FEATURES / TRUE_NUM_GROUPS] * TRUE_NUM_GROUPS
    EXPERT_KNOWLEDGE_GROUP_FEATURE_SIZES = [TOTAL_FEATURES / NUM_GROUPS] * NUM_GROUPS

    COARSE_LAMBDAS = [1, 1e-1]
    if RUN_HC_POOLED:
        print "RUN POOLED FOR GS and HC"
    else:
        print "UNPOOLED VS. POOLED"

    seed = np.random.randint(0, 1e5)
    np.random.seed(seed)
    print "RANDOM SEED", seed

    def _hillclimb_coarse_grid_search(optimization_func, *args, **kwargs):
        # Allow multiple initializations for gradient descent
        start_time = time.time()
        best_cost = 1e10
        best_beta = []
        best_cost_path = []
        best_lambda = 0
        for init_lambda in COARSE_LAMBDAS:
            kwargs["initial_lambda"] = init_lambda
            beta_guess, cost_path = optimization_func(*args, **kwargs)
            if best_cost > cost_path[-1]:
                best_cost = cost_path[-1]
                best_cost_path = cost_path
                best_beta = beta_guess
                best_lambda = init_lambda
        end_time = time.time()
        return best_beta, best_cost_path, end_time - start_time

    hc_results = MethodResults(HC_GROUPED_LASSO_LABEL)
    hc_pooled_results = MethodResults(HC_GROUPED_LASSO_LABEL + "_POOLED")
    gs_results = MethodResults(GS_GROUPED_LASSO_LABEL)

    for i in range(0, NUM_RUNS):
        # Generate data
        beta_reals, X_train, y_train, X_validate, y_validate, X_test, y_test = sparse_groups(TRAIN_SIZE, TRUE_GROUP_FEATURE_SIZES)

        def _create_method_result(beta_guesses, runtime):
            # Helper method for aggregating results
            test_err = testerror_grouped(X_test, y_test, beta_guesses)
            validation_err = testerror_grouped(X_validate, y_validate, beta_guesses)
            beta_guesses_all = np.concatenate(beta_guesses)
            beta_reals_all = np.concatenate(beta_reals)
            beta_err = betaerror(beta_reals_all, beta_guesses_all)
            guessed_nonzero_elems = np.where(get_nonzero_indices(beta_guesses_all, threshold=ZERO_THRESHOLD))
            true_nonzero_elems = np.where(get_nonzero_indices(beta_reals_all, threshold=ZERO_THRESHOLD))
            intersection = np.intersect1d(np.array(guessed_nonzero_elems), np.array(true_nonzero_elems))
            sensitivity = intersection.size / float(guessed_nonzero_elems[0].size) * 100
            return MethodResult(test_err=test_err, validation_err=validation_err, beta_err=beta_err, sensitivity=sensitivity, runtime=runtime)

        # Run gradient descent
        if RUN_HC_POOLED:
            hc_pooled_beta_guesses, hc_pooled_costpath, runtime = _hillclimb_coarse_grid_search(
                hc_pooled.run, X_train, y_train, X_validate, y_validate, EXPERT_KNOWLEDGE_GROUP_FEATURE_SIZES
            )
            hc_pooled_results.append(_create_method_result(hc_pooled_beta_guesses, runtime))
        else:
            hc_beta_guesses, hc_costpath, runtime = _hillclimb_coarse_grid_search(
                hc.run, X_train, y_train, X_validate, y_validate, EXPERT_KNOWLEDGE_GROUP_FEATURE_SIZES
            )
            hc_results.append(_create_method_result(hc_beta_guesses, runtime))

        # Run grid search
        start = time.time()
        gs_beta_guesses, gs_lowest_cost = gridsearch_grouped_lasso.run(X_train, y_train, X_validate, y_validate, EXPERT_KNOWLEDGE_GROUP_FEATURE_SIZES)
        runtime = time.time() - start
        gs_results.append(_create_method_result(gs_beta_guesses, runtime))

        print "===========RUN %d ============" % i
        if RUN_HC_POOLED:
            hc_pooled_results.print_results()
        else:
            hc_results.print_results()
        gs_results.print_results()

if __name__ == "__main__":
    main(sys.argv[1:])
