import time
import numpy as np
from scipy.optimize import minimize
from fitted_model import Fitted_Model

class Grid_Search:
    MAX_COST = 1e5

    def __init__(self, data, settings=None):
        self.data = data
        self.settings = settings
        self._create_problem_wrapper()

    def run(self, lambdas1, lambdas2=None, log_file=None):
        self.log_file = log_file
        start = time.time()

        best_cost = self.MAX_COST
        # if only one lambda to tune
        if lambdas2 is None:
            self.fmodel = Fitted_Model(num_lambdas=1)
            self.log("%s lambda values: %s" % (self.method_label, lambdas1))
            for l1 in lambdas1:
                curr_lambdas = np.array([l1])
                model_params = self.problem_wrapper.solve(curr_lambdas)
                cost = self.get_validation_cost(model_params)

                if best_cost > cost:
                    best_cost = cost
                    self.log("%s: best_validation_error %f" % (self.method_label, best_cost))
                    self.fmodel.update(curr_lambdas, model_params, cost)
        else:
            self.fmodel = Fitted_Model(num_lambdas=2)
            self.log("%s lambda1 values: %s" % (self.method_label, lambdas1))
            self.log("%s lambda2 values: %s" % (self.method_label, lambdas2))
            # if two lambdas to tune
            for l1 in lambdas1:
                for l2 in lambdas2:
                    curr_lambdas = np.array([l1,l2])
                    model_params = self.problem_wrapper.solve(curr_lambdas)
                    cost = self.get_validation_cost(model_params)

                    if best_cost > cost:
                        best_cost = cost
                        self.log("%s: best_validation_error %f" % (self.method_label, best_cost))
                        self.fmodel.update(curr_lambdas, model_params, cost)

        runtime = time.time() - start
        self.fmodel.set_runtime(runtime)

        self.log("%s: best cost %f, lambda %s" % (self.method_label, best_cost, self.fmodel.current_lambdas))

    def log(self, log_str):
        if self.log_file is None:
            print log_str
        else:
            self.log_file.write("%s\n" % log_str)
