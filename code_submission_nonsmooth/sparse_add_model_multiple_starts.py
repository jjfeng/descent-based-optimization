import getopt
import time
import sys
import traceback
import pickle
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt

from sparse_add_models_hillclimb import Sparse_Add_Model_Hillclimb, Sparse_Add_Model_Hillclimb_Simple
from sparse_add_models_neldermead import Sparse_Add_Model_Nelder_Mead, Sparse_Add_Model_Nelder_Mead_Simple
from data_generator import DataGenerator
from method_results import MethodResults
from method_results import MethodResult
from fitted_model import Fitted_Model
from iteration_models import Simulation_Settings, Iteration_Data

from common import *

########
# FUNCTIONS FOR ADDITIVE MODEL
########
def identity_fcn(x):
    return x.reshape(x.size, 1)

def big_sin(x):
    return identity_fcn(9 * np.sin(x*3))

def big_cos_sin(x):
    return identity_fcn(6 * (np.cos(x * 1.25) + np.sin(x/2 + 0.5)))

def crazy_down_sin(x):
    return identity_fcn(x * np.sin(x) - x)

def pwr_small(x):
    return identity_fcn(np.power(x/2,2) - 10)

def const_zero(x):
    return np.zeros(x.shape)

########
# Model settings
########
class Sparse_Add_Models_Multiple_Starts_Settings(Simulation_Settings):
    results_folder = "results/sparse_add_models_multiple_starts"
    num_funcs = 3
    num_zero_funcs = 12
    train_size = 60
    validate_size = 30
    test_size = 30
    init_size = 30
    smooth_fcns = [big_sin, identity_fcn, big_cos_sin, crazy_down_sin, pwr_small]
    plot = False
    min_init_log_lambda = -2
    max_init_log_lambda = 1
    big_init_factor = 5
    method = "HC"

    def print_settings(self):
        print "SETTINGS"
        obj_str = "method %s\n" % self.method
        obj_str += "num_funcs %d\n" % self.num_funcs
        obj_str += "num_zero_funcs %d\n" % self.num_zero_funcs
        obj_str += "t/v/t size %d/%d/%d\n" % (self.train_size, self.validate_size, self.test_size)
        obj_str += "snr %f\n" % self.snr
        obj_str += "nm_iters %d\n" % self.nm_iters
        obj_str += "init_size (random) %d\n" % self.init_size
        print obj_str

class CumulativeInitializationResults:
    def __init__(self, data, settings):
        self.settings = settings
        self.data = data
        self.lambdas = []
        self.lambda_val_cost = []
        self.lambda_test_cost = []
        self.cumulative_best_lambda = []
        self.cumulative_val_cost = []
        self.cumulative_test_cost = []

    def update(self, method_res):
        val_err = method_res.stats["validation_err"]
        test_err = method_res.stats["test_err"]
        lambdas = method_res.lambdas
        self.lambdas.append(lambdas)
        self.lambda_test_cost.append(test_err)
        self.lambda_val_cost.append(val_err)
        if len(self.cumulative_val_cost) == 0 or self.cumulative_val_cost[-1] > val_err:
            self.cumulative_best_lambda.append(lambdas)
            self.cumulative_val_cost.append(val_err)
            self.cumulative_test_cost.append(test_err)
        else:
            self.cumulative_best_lambda.append(self.cumulative_best_lambda[-1])
            self.cumulative_val_cost.append(self.cumulative_val_cost[-1])
            self.cumulative_test_cost.append(self.cumulative_test_cost[-1])

#########
# MAIN FUNCTION
#########
def main(argv):
    num_threads = 1
    seed = 20
    print "seed", seed
    np.random.seed(seed)

    try:
        opts, args = getopt.getopt(argv,"t:r:f:z:a:b:c:s:m:i:")
    except getopt.GetoptError:
        sys.exit(2)

    settings = Sparse_Add_Models_Multiple_Starts_Settings()
    for opt, arg in opts:
        if opt == '-t':
            num_threads = int(arg)
        elif opt == '-f':
            settings.num_funcs = int(arg)
        elif opt == '-z':
            settings.num_zero_funcs = int(arg)
        elif opt == '-a':
            settings.train_size = int(arg)
        elif opt == '-b':
            settings.validate_size = int(arg)
        elif opt == '-c':
            settings.test_size = int(arg)
        elif opt == "-s":
            settings.snr = float(arg)
        elif opt == "-m":
            assert(arg in ["HC", "NM"])
            settings.method = arg
        elif opt == "-i":
            settings.init_size = int(arg)

    assert(settings.num_funcs <= len(settings.smooth_fcns))

    smooth_fcn_list = settings.smooth_fcns[:settings.num_funcs] + [const_zero] * settings.num_zero_funcs
    data_gen = DataGenerator(settings)

    observed_data = data_gen.make_additive_smooth_data(smooth_fcn_list)

    # Create initial lambdas
    num_lambdas = 1 + settings.num_funcs + settings.num_zero_funcs
    # initial_lambdas_set = [
    #     np.array([10] + [1] * (num_lambdas - 1)),
    #     np.array([0.1] + [0.01] * (num_lambdas - 1)),
    # ]
    # for i in range(settings.init_size - 2):
    #     init_l = np.power(10.0, np.random.randint(low=settings.min_init_log_lambda, high=settings.max_init_log_lambda, size=num_lambdas))

    # Pool the last lambdas together. Shuffle the possibilities
    initial_lambdas_set = []
    init_values = np.power(10.0, np.arange(settings.min_init_log_lambda, settings.max_init_log_lambda + 1))
    for l1 in init_values:
        for l2 in init_values:
            full_init_l = np.array([l1] + [l2] * (num_lambdas - 1))
            initial_lambdas_set.append(full_init_l)
    permute_idxs = np.random.permutation(np.arange(0, len(init_values) * len(init_values)))
    settings.init_size = permute_idxs.size

    settings.print_settings()
    sys.stdout.flush()

    run_data = []
    for i, idx in enumerate(permute_idxs):
        init_lambdas = initial_lambdas_set[idx]
        print "init_lambdas", init_lambdas
        run_data.append(Iteration_Data(i, observed_data, settings, init_lambdas=[init_lambdas]))

    if num_threads > 1:
        print "Do multiprocessing"
        pool = Pool(num_threads)
        results = pool.map(fit_data_for_iter_safe, run_data)
    else:
        print "Avoiding multiprocessing"
        results = map(fit_data_for_iter_safe, run_data)

    print "results", results
    cum_results = CumulativeInitializationResults(observed_data, settings)
    for r in results:
        print r
        cum_results.update(r)

    print "==========RUNS============"
    print "cumulative_val_cost", cum_results.cumulative_val_cost
    print "cumulative_test_cost", cum_results.cumulative_test_cost

    pickle_file_name = "%s/tmp/%s_many_inits_%d_%d_%d_%d_%d_%d_%d.pkl" % (
        settings.results_folder,
        settings.method,
        settings.num_funcs,
        settings.num_zero_funcs,
        settings.train_size,
        settings.validate_size,
        settings.test_size,
        settings.snr,
        settings.init_size,
    )
    print "pickle_file_name", pickle_file_name
    with open(pickle_file_name, "wb") as f:
        pickle.dump({
            "initial_lambdas_set": initial_lambdas_set,
            "cum_results": cum_results,
         }, f)

    # plot_mult_inits(cum_results, str_identifer)
    print "DONE!"

def fit_data_for_iter_safe(iter_data):
    result = None
    try:
        result = fit_data_for_iter(iter_data)
    except Exception as e:
        print "Exception caught in iter %d: %s" % (iter_data.i, e)
        traceback.print_exc()
    return result

def fit_data_for_iter(iter_data):
    data = iter_data.data
    settings = iter_data.settings
    method = settings.method

    str_identifer = "%s_many_inits_%d_%d_%d_%d_%d_%d_%d_%d" % (
        settings.method,
        settings.num_funcs,
        settings.num_zero_funcs,
        settings.train_size,
        settings.validate_size,
        settings.test_size,
        settings.snr,
        settings.init_size,
        iter_data.i,
    )

    log_file_name = "%s/tmp/log_%s.txt" % (settings.results_folder, str_identifer)
    print "log_file_name", log_file_name
    print "iter_data", iter_data.init_lambdas
    # set file buffer to zero so we can see progress
    with open(log_file_name, "w", buffering=0) as f:
        if method == "NM":
            algo = Sparse_Add_Model_Nelder_Mead(data)
            algo.run(iter_data.init_lambdas, num_iters=settings.nm_iters, log_file=f)
            return create_method_result(data, algo.fmodel)
        elif method == "HC":
            algo = Sparse_Add_Model_Hillclimb(data)
            algo.run(iter_data.init_lambdas, debug=False, log_file=f)
            return create_method_result(data, algo.fmodel)

def create_method_result(data, fmodel):
    test_err = testerror_sparse_add_smooth(
        data.y_test,
        data.test_idx,
        fmodel.best_model_params
    )
    print "test_err", test_err
    return MethodResult({
            "test_err":test_err,
            "validation_err":fmodel.best_cost,
        },
        lambdas=fmodel.best_lambdas,
    )

def plot_mult_inits(str_identifer):
    # Plot how the validation error and test error change as number of initializations change

    with open("results/sparse_add_models_multiple_starts/tmp/HC_%s.pkl" % str_identifer) as f:
        results_hc = pickle.load(f)
        cum_results_hc = results_hc["cum_results"]
    with open("results/sparse_add_models_multiple_starts/tmp/NM_%s.pkl" % str_identifer) as f:
        results_nm = pickle.load(f)
        cum_results_nm = results_nm["cum_results"]

    file_name = "results/sparse_add_models_multiple_starts/figures/%s" % str_identifer

    plt.clf()
    range_max = len(cum_results_nm.cumulative_test_cost)
    plt.plot(
        range(1, range_max + 1),
        cum_results_nm.cumulative_val_cost,
        label="NM Validation error",
        color="red",
        marker="^",
    )
    plt.plot(
        range(1, range_max + 1),
        cum_results_nm.cumulative_test_cost,
        label="NM Test error",
        color="red",
        linestyle="--",
        marker="^",
    )
    plt.plot(
        range(1, range_max + 1),
        cum_results_hc.cumulative_val_cost,
        label="HC Validation error",
        color="green",
        marker="o",
    )
    plt.plot(
        range(1, range_max + 1),
        cum_results_hc.cumulative_test_cost,
        label="HC Test error",
        color="green",
        linestyle="--",
        marker="o",
    )
    plt.xlim(1, range_max)
    plt.xlabel("Number of Initializations")
    plt.ylabel("Error")
    plt.legend()
    figname = "%s_trend.png" % file_name
    print "figname", figname
    plt.savefig(figname)

    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 6), sharey=True)
    axes[0].boxplot(
        [
            cum_results_hc.lambda_val_cost,
            cum_results_nm.lambda_val_cost,
        ]
    , labels=["HC", "NM"])
    axes[0].set_title('Validation Error')
    axes[1].boxplot(
        [
            cum_results_hc.lambda_test_cost,
            cum_results_nm.lambda_test_cost,
        ]
    , labels=["HC", "NM"])
    axes[1].set_title('Test Error')
    figname = "%s_box.png" % file_name
    print "figname", figname
    plt.savefig(figname)

if __name__ == "__main__":
    main(sys.argv[1:])
