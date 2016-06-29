import sys
import scipy as sp
from scipy.sparse.linalg import spsolve
import numpy as np
import cvxpy

from common import *
from convexopt_solvers import GenAddModelProblemWrapper

class GenAddModelHillclimb:
    NUMBER_OF_ITERATIONS = 40
    BOUNDARY_FACTOR = 0.999
    STEP_SIZE = 1
    LAMBDA_MIN = 1e-6
    SHRINK_MIN = 1e-5
    SHRINK_SHRINK_FACTOR = 0.1
    SHRINK_FACTOR_INIT = 1
    DECREASING_ENOUGH_THRESHOLD = 1e-4 * 5
    BACKTRACK_ALPHA = 0.01
    USE_BOUNDARY = True

    @staticmethod
    def stack(data_tuple):
        # Given a tuple of data matrices, returns the combined data and the indices
        # for which group the observation belonged to
        stacked_data = np.vstack(data_tuple)
        res = [stacked_data]
        start_idx = 0
        for d in data_tuple:
            res.append(np.arange(start_idx, start_idx + d.shape[0]))
            start_idx += d.shape[0]
        return res

    def __init__(self, X_train, y_train, X_validate, y_validate, X_test):
        self.method_label = "hc_gam"
        self.X_train = X_train
        self.y_train = y_train
        self.X_validate = X_validate
        self.y_validate = y_validate
        self.X_test = X_test
        self.X_full, self.train_idx, self.validate_idx, self.test_idx = self.stack((X_train, X_validate, X_test))

        self.problem_wrapper = GenAddModelProblemWrapper(self.X_full, self.train_idx, self.y_train)
        self.num_samples = self.problem_wrapper.num_samples
        self.num_features = self.problem_wrapper.num_features

        self.train_indices = self.problem_wrapper.train_indices
        self.M = self.problem_wrapper.train_identifier
        num_validate = len(self.validate_idx)
        num_train = len(self.train_idx)
        self.validate_M = np.matrix(np.zeros((num_validate, self.num_samples)))
        self.validate_M[np.arange(num_validate), self.validate_idx] = 1

        self.MM = self.M.T * self.M/num_train
        self.DD = []
        for feat_idx in range(self.num_features):
            D = self.problem_wrapper.diff_matrices[feat_idx]
            self.DD.append(D.T * D/self.num_samples)

        self.cost_fcn = testerror_multi_smooth

    def run(self, initial_lambdas):
        curr_regularization = initial_lambdas

        thetas = self.problem_wrapper.solve(curr_regularization)
        assert(thetas is not None)

        current_cost = self.cost_fcn(self.y_validate, self.validate_idx, thetas)

        # track progression
        cost_path = [current_cost]

        method_step_size = self.STEP_SIZE
        shrink_factor = self.SHRINK_FACTOR_INIT
        potential_thetas = None
        for i in range(0, self.NUMBER_OF_ITERATIONS):
            print "ITER", i, "current lambdas", curr_regularization
            lambda_derivatives = self._get_lambda_derivatives(curr_regularization, thetas)
            assert(not np.any(np.isnan(lambda_derivatives)))

            potential_new_regularization = self._get_updated_lambdas(
                curr_regularization,
                shrink_factor * method_step_size,
                lambda_derivatives
            )
            try:
                potential_thetas = self.problem_wrapper.solve(potential_new_regularization)
            except cvxpy.error.SolverError:
                potential_thetas = None

            if potential_thetas is None:
                potential_cost = current_cost * 100
            else:
                potential_cost = self.cost_fcn(self.y_validate, self.validate_idx, potential_thetas)

            raw_backtrack_check = current_cost - self.BACKTRACK_ALPHA * shrink_factor * method_step_size * np.linalg.norm(lambda_derivatives)**2
            backtrack_check = current_cost if raw_backtrack_check < 0 else raw_backtrack_check
            while potential_cost >= backtrack_check and shrink_factor > self.SHRINK_MIN:
                # shrink step size
                if potential_cost > 2 * current_cost:
                    shrink_factor *= self.SHRINK_SHRINK_FACTOR * 0.01
                else:
                    shrink_factor *= self.SHRINK_SHRINK_FACTOR

                potential_new_regularization = self._get_updated_lambdas(curr_regularization, shrink_factor * method_step_size, lambda_derivatives)
                try:
                    potential_thetas = self.problem_wrapper.solve(potential_new_regularization)
                except cvxpy.error.SolverError as e:
                    # cvxpy could not find a soln
                    potential_thetas = None

                if potential_thetas is None:
                    potential_cost = current_cost * 100
                else:
                    potential_cost = self.cost_fcn(self.y_validate, self.validate_idx, potential_thetas)

                raw_backtrack_check = current_cost - self.BACKTRACK_ALPHA * shrink_factor * method_step_size * np.linalg.norm(lambda_derivatives)**2
                backtrack_check = current_cost if raw_backtrack_check < 0 else raw_backtrack_check

            # track progression
            if cost_path[-1] < potential_cost:
                # cost is increasing
                break
            else:
                curr_regularization = potential_new_regularization
                current_cost = potential_cost
                thetas = potential_thetas
                cost_path.append(current_cost)

                if cost_path[-2] - cost_path[-1] < self.DECREASING_ENOUGH_THRESHOLD:
                    # progress too slow
                    break

            if shrink_factor < self.SHRINK_MIN:
                # shrink size too small
                break

        return thetas, cost_path, curr_regularization

    def _get_lambda_derivatives(self, curr_lambdas, curr_thetas):
        H = np.tile(self.MM, (self.num_features, self.num_features))
        num_feat_sam = self.num_features * self.num_samples

        H += sp.linalg.block_diag(*[
            curr_lambdas[i] * self.DD[i] for i in range(self.num_features)
        ])
        H = sp.sparse.csr_matrix(H)

        sum_thetas = np.matrix(np.sum(curr_thetas, axis=1))
        dloss_dlambdas = []
        num_validate = self.y_validate.size
        for i in range(self.num_features):
            b = np.zeros((num_feat_sam, 1))
            b[i * self.num_samples:(i + 1) * self.num_samples, :] = -self.DD[i] * curr_thetas[:,i]
            dtheta_dlambdai = spsolve(H, b)
            dtheta_dlambdai = dtheta_dlambdai.reshape((self.num_features, self.num_samples)).T
            sum_dtheta_dlambdai = np.matrix(np.sum(dtheta_dlambdai, axis=1)).T
            dloss_dlambdai = -1.0/num_validate * sum_dtheta_dlambdai[self.validate_idx].T * (self.y_validate - sum_thetas[self.validate_idx])
            dloss_dlambdas.append(dloss_dlambdai[0,0])

        return np.array(dloss_dlambdas)

    def _get_updated_lambdas(self, lambdas, method_step_size, lambda_derivatives):
        new_step_size = method_step_size
        if self.USE_BOUNDARY:
            # Do not allow super tiny lambdas
            potential_lambdas = lambdas - method_step_size * lambda_derivatives

            for idx in range(0, lambdas.size):
                if lambdas[idx] > self.LAMBDA_MIN and potential_lambdas[idx] < (1 - self.BOUNDARY_FACTOR) * lambdas[idx]:
                    smaller_step_size = self.BOUNDARY_FACTOR * lambdas[idx] / lambda_derivatives[idx]
                    new_step_size = min(new_step_size, smaller_step_size)

        return np.maximum(lambdas - new_step_size * lambda_derivatives, self.LAMBDA_MIN)
