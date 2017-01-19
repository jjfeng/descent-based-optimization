# Lasso simulation to show that the best lasso parameters to minimize the validation
# error are not at the knots along the lasso path.

import time
import sys
import pickle
import numpy as np
from multiprocessing import Pool

from iteration_models import Simulation_Settings, Iteration_Data
from method_results import MethodResult
from convexopt_solvers import LassoProblemWrapper

from data_generator import DataGenerator

from common import *
from sklearn import linear_model
import matplotlib.pyplot as plt

class Lasso_Settings(Simulation_Settings):
    results_folder = "results/lasso"
    num_features = 50
    num_nonzero_features = 3
    train_size = 15
    validate_size = 10
    test_size = 40
    method = "HC"


def plot_lasso_path(alphas, coefs):
    # alphas is the lasso param
    plt.figure()
    neg_log_alphas = -np.log10(alphas)
    for coef in coefs:
        plt.plot(neg_log_alphas, coef)
    plt.xlabel("new log lasso param")
    plt.ylabel("coef value")
    plt.show()

def get_dist_of_closest_lambda(lam, lambda_path):
    lambda_knot_dists = np.abs(lambda_path - lam)
    print "lambda_knot_dists", lambda_knot_dists
    min_idx = np.argmin(lambda_knot_dists)
    return lambda_knot_dists[min_idx], min_idx

def do_lasso_simulation(data):
    # Make lasso path
    lasso_path, coefs, _ = linear_model.lasso_path(
        data.X_train,
        np.array(data.y_train.flatten().tolist()[0]), # reshape appropriately
        method='lasso'
    )
    prob = LassoProblemWrapper(
        data.X_train,
        data.y_train
    )

    min_log10 = np.log10(lasso_path[-1]) - 0.01
    max_log10 = np.log10(lasso_path[0]) + 0.01
    gs_lambdas = np.power(10, np.arange(min_log10, max_log10, (max_log10 - min_log10)/len(lasso_path)/5))
    best_l = gs_lambdas[0]
    best_val_error = 10000
    best_beta = 0
    for l in gs_lambdas:
        beta = prob.solve(np.array([l]))
        val_error = testerror_lasso(data.X_validate, data.y_validate, beta)
        if best_val_error > val_error:
            best_val_error = val_error
            best_l = l
            best_beta = beta

    min_dist, idx = get_dist_of_closest_lambda(best_l, lasso_path)
    print "min_dist", min_dist
    return min_dist

def plot_min_dists():
    figure_file_name = "results/lasso/lasso_knot_locations.png"
    with open("results/lasso/lasso_knot_locations.pkl", "r") as f:
        min_dists = pickle.load(f)
    plt.hist(min_dists, bins=np.logspace(np.log(np.min(min_dists)), np.log(np.max(min_dists)), 50))
    plt.gca().set_xscale("log")
    plt.xlim(1e-6, 1e-2)
    plt.xlabel("Distance Between $\hat{\lambda}$ and Closest Knot")
    plt.ylabel("Frequency")
    print "figure_file_name", figure_file_name
    plt.savefig(figure_file_name)

np.random.seed(10)
NUM_RUNS = 500
num_threads = 6

settings = Lasso_Settings()
data_gen = DataGenerator(settings)
initial_lambdas_set = [np.ones(1) * 0.1]

# Make data
datas = []
for i in range(NUM_RUNS):
    datas.append(
        data_gen.make_simple_linear(settings.num_features, settings.num_nonzero_features)
    )

pool = Pool(num_threads)
min_dists = pool.map(do_lasso_simulation, datas)

pickle_file_name = "%s/lasso_knot_locations.pkl" % settings.results_folder
print "pickle_file_name", pickle_file_name
with open(pickle_file_name, "wb") as f:
    pickle.dump(min_dists, f)

mean_best_l_dists = np.mean(min_dists)
print "Mean distance between the best lasso vs. lasso path knots", mean_best_l_dists

plot_min_dists()
