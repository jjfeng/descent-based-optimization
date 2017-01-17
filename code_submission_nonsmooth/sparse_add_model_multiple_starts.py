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
    num_zero_funcs = 20
    init_size = 30
    smooth_fcns = [big_sin, identity_fcn, big_cos_sin, crazy_down_sin, pwr_small]
    plot = False
    min_init_log_lambda = -2
    max_init_log_lambda = 1
    big_init_factor = 5
    method = "HC"
    method_result_keys = [
        "test_err",
        "validation_err",
        "runtime",
        "num_solves",
        "perc_nonzero_true_f", # among true nonzero f, what percent correct
        "perc_zero_true_f", # among true zero f, what percent correct
        "perc_nonzero_f", # among guessed nonzero f, what percent correct
        "perc_zero_f", # among true zero f, what percent correct
    ]

    def print_settings(self):
        print "SETTINGS"
        obj_str = "method %s\n" % self.method
        obj_str += "num_funcs %d\n" % self.num_funcs
        obj_str += "num_zero_funcs %d\n" % self.num_zero_funcs
        obj_str += "t/v/t size %d/%d/%d\n" % (self.train_size, self.validate_size, self.test_size)
        obj_str += "snr %f\n" % self.snr
        obj_str += "sp runs %d\n" % self.spearmint_numruns
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
        self.cumulative_best_model = []
        self.cumulative_best_lambda = []
        self.cumulative_val_cost = []
        self.cumulative_test_cost = []

    def update(self, fmodel):
        test_err = testerror_sparse_add_smooth(
            self.data.y_test,
            self.data.test_idx,
            fmodel.best_model_params
        )
        self.lambdas.append(fmodel.best_lambdas)
        self.lambda_test_cost.append(test_err)
        self.lambda_val_cost.append(fmodel.best_cost)
        if len(self.cumulative_val_cost) == 0 or self.cumulative_val_cost[-1] > fmodel.best_cost:
            self.cumulative_best_model.append(fmodel.best_model_params)
            self.cumulative_best_lambda.append(fmodel.best_lambdas)
            self.cumulative_val_cost.append(fmodel.best_cost)
            self.cumulative_test_cost.append(test_err)
        else:
            self.cumulative_best_model.append(self.cumulative_best_model[-1])
            self.cumulative_best_lambda.append(self.cumulative_best_lambda[-1])
            self.cumulative_val_cost.append(self.cumulative_val_cost[-1])
            self.cumulative_test_cost.append(self.cumulative_test_cost[-1])

#########
# MAIN FUNCTION
#########
def main(argv):
    num_runs = 1
    seed = 20
    print "seed", seed
    np.random.seed(seed)

    try:
        opts, args = getopt.getopt(argv,"r:f:z:a:b:c:s:m:i:")
    except getopt.GetoptError:
        sys.exit(2)

    settings = Sparse_Add_Models_Multiple_Starts_Settings()
    for opt, arg in opts:
        if opt == '-r':
            num_runs = int(arg)
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
            assert(arg in METHODS)
            settings.method = arg
        elif opt == "-i":
            settings.init_size = int(arg)

    settings.print_settings()
    sys.stdout.flush()

    assert(settings.num_funcs <= len(settings.smooth_fcns))
    smooth_fcn_list = settings.smooth_fcns[:settings.num_funcs] + [const_zero] * settings.num_zero_funcs
    data_gen = DataGenerator(settings)

    for i in range(num_runs):
        print "fit iter %d" % i
        str_identifer = "HC_many_inits_%d_%d_%d_%d_%d_%d_%d_%d" % (
            settings.num_funcs,
            settings.num_zero_funcs,
            settings.train_size,
            settings.validate_size,
            settings.test_size,
            settings.snr,
            settings.init_size,
            i,
        )
        observed_data = data_gen.make_additive_smooth_data(smooth_fcn_list)
        fit_data_for_iter(observed_data, settings, str_identifer)

    print "DONE!"

def fit_data_for_iter(data, settings, str_identifer):
    num_lambdas = 1 + settings.num_funcs + settings.num_zero_funcs
    initial_lambdas_set = [
        np.array([10] + [1] * (num_lambdas - 1)),
        np.array([0.1] + [0.01] * (num_lambdas - 1)),
    ]
    for i in range(settings.init_size - 2):
        init_l = np.power(10.0, np.random.randint(low=settings.min_init_log_lambda, high=settings.max_init_log_lambda, size=num_lambdas))
        initial_lambdas_set.append(init_l)

    method = settings.method

    log_file_name = "%s/tmp/log_%s.txt" % (settings.results_folder, str_identifer)
    print "log_file_name", log_file_name
    # set file buffer to zero so we can see progress
    cum_results = CumulativeInitializationResults(data, settings)
    with open(log_file_name, "w", buffering=0) as f:
        algo = Sparse_Add_Model_Hillclimb(data)
        cumulative_model = None
        for init_idx, init_lams in enumerate(initial_lambdas_set):
            algo.run([init_lams], debug=False, log_file=f)
            cum_results.update(algo.fmodel)

    pickle_file_name = "%s/tmp/%s.pkl" % (settings.results_folder, str_identifer)
    print "pickle_file_name", pickle_file_name
    with open(pickle_file_name, "wb") as f:
        pickle.dump({
            "initial_lambdas_set": initial_lambdas_set,
            "cum_results": cum_results,
         }, f)

    plot_mult_inits(cum_results, str_identifer)

def get_which_funcs_zero(thetas, thres=1e-4):
    num_funcs = thetas.shape[1]
    num_points = thetas.shape[0]
    return np.array([get_norm2(thetas[:,i])/np.sqrt(num_points) < thres for i in range(num_funcs)])

def plot_mult_inits(cum_results, str_identifer, label=None):
    # Plot how the validation error and test error change as number of initializations change

    file_name = "%s/figures/%s" % (cum_results.settings.results_folder, str_identifer)

    plt.clf()
    # plt.plot(
    #     range(cum_results.settings.init_size),
    #     cum_results.lambda_val_cost,
    #     label="Validation error",
    #     color="green",
    #     linestyle="--",
    # )
    plt.plot(
        range(cum_results.settings.init_size),
        cum_results.cumulative_val_cost,
        label="Validation error",
        color="green",
    )
    # plt.plot(
    #     range(cum_results.settings.init_size),
    #     cum_results.lambda_test_cost,
    #     label="Test error",
    #     color="red",
    #     linestyle="--",
    # )
    plt.plot(
        range(cum_results.settings.init_size),
        cum_results.cumulative_test_cost,
        label="Test error",
        color="red",
    )
    plt.xlabel("Number of Initializations")
    plt.ylabel("Error")
    plt.legend()
    figname = "%s.png" % file_name
    print "figname", figname
    plt.savefig(figname)

if __name__ == "__main__":
    main(sys.argv[1:])
