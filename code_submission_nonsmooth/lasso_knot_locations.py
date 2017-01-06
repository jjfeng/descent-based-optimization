# Lasso simulation to show that the best lasso parameters to minimize the validation
# error are not at the knots along the lasso path.

import time
import sys
import numpy as np

from iteration_models import Simulation_Settings, Iteration_Data
from lasso_hillclimb import Lasso_Hillclimb
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
    lambda_knot_dists = np.abs(lasso_path - lam)
    min_idx = np.argmin(lambda_knot_dists)
    return lambda_knot_dists[min_idx], min_idx

np.random.seed(10)
NUM_RUNS = 30

settings = Lasso_Settings()
data_gen = DataGenerator(settings)
initial_lambdas_set = [np.ones(1) * 0.1]

# Make data
best_l_dists = 0

for i in range(NUM_RUNS):
    data = data_gen.make_correlated(settings.num_features, settings.num_nonzero_features)

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

    gs_lambdas = np.power(10, np.arange(np.log10(lasso_path[-1]), np.log10(lasso_path[0]), 0.005))
    print "gs_lambdas", gs_lambdas[0], gs_lambdas[-1]
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
    best_l_dists += min_dist

    print "best_l", best_l
    print "lasso_path", lasso_path[idx]


mean_best_l_dists = best_l_dists/NUM_RUNS
print "Mean distance between the best lasso vs. lasso path knots", mean_best_l_dists
