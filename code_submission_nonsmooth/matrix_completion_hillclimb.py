from cvxpy import *
import cvxopt
import numpy as np
import scipy as sp
from common import testerror_matrix_completion
from gradient_descent_algo import Gradient_Descent_Algo
from convexopt_solvers import MatrixCompletionProblemWrapperSimple


class Matrix_Completion_Hillclimb_Base(Gradient_Descent_Algo):
    def _create_descent_settings(self):
        self.num_iters = 20
        self.step_size_init = 1
        self.step_size_min = 1e-6
        self.shrink_factor = 0.1
        self.decr_enough_threshold = 1e-4 * 5
        self.use_boundary = False
        self.boundary_factor = 0.999999
        self.backtrack_alpha = 0.001

        self.train_vec_diag = self._get_train_mask()
        self.onesT_row = np.matrix(np.ones(self.data.num_rows))
        self.onesT_col = np.matrix(np.ones(self.data.num_cols))

    def get_validate_cost(self, model_params):
        return testerror_matrix_completion(
            self.data,
            self.data.validate_idx,
            model_params
        )

class Matrix_Completion_Hillclimb_Simple(Matrix_Completion_Hillclimb_Base):
    method_label = "Matrix_Completion_Hillclimb"

    def _create_lambda_configs(self):
        self.lambda_mins = [1e-6, 1e-6]

    def _create_problem_wrapper(self):
        self.problem_wrapper = MatrixCompletionProblemWrapperSimple(self.data)

    def _make_flat(self, m):
        return np.reshape(m, (m.size, 1), order='F')

    def _get_lambda_derivatives(self):
        alpha = self.fmodel.current_model_params["row_theta"]
        beta = self.fmodel.current_model_params["col_theta"]
        gamma = self.fmodel.current_model_params["interaction_m"]

        print "alpha", alpha
        print "Beta", beta

        # self._check_optimality_conditions_simple(alpha, beta)
        self._check_optimality_conditions(alpha, beta, gamma)

        # self._get_gradient_lambda1(alpha, beta, self.fmodel.current_lambdas)
        self._get_gradient_lambda0(alpha, beta, gamma, self.fmodel.current_lambdas)

    def _check_optimality_conditions_simple(self, alpha, beta):
        num_train = self.data.train_idx.size

        grad_at_opt = [
            -1.0/num_train * (self.train_vec_diag * self._make_flat(
                self.data.observed_matrix -
                self.data.row_features * alpha * self.onesT_row -
                (self.data.col_features * beta * self.onesT_col).T
            )).T * self._make_flat(self.data.row_features[:,i] * self.onesT_row)
            + self.fmodel.current_lambdas[1] * (np.sign(alpha[i]) + alpha[i])
            for i in range(self.data.num_row_features)
        ]
        print "grad_at_opt", grad_at_opt
        assert(np.all(np.abs(grad_at_opt) < 1e-2))

    def _check_optimality_conditions(self, alpha, beta, gamma):
        u, s, v = np.linalg.svd(gamma)
        u_hat = u
        sigma_hat = np.diag(s)
        print "sigma_hat", sigma_hat
        v_hat = v.T

        num_train = self.data.train_idx.size

        train_mask = np.matrix(np.zeros((self.data.num_rows, self.data.num_cols))).T
        train_vec_mask = train_mask.flatten('F').A[0]
        train_vec_mask[self.data.train_idx] = 1
        train_vec_diag = np.diag(train_vec_mask)
        print "train_vec_diag", train_vec_diag.shape

        onesT_row = np.matrix(np.ones(self.data.num_rows))
        onesT_col = np.matrix(np.ones(self.data.num_cols))

        d_square_loss = -1.0/num_train * train_vec_diag * np.reshape(
            self.data.observed_matrix -
            self.data.row_features * alpha * onesT_row -
            (self.data.col_features * beta * onesT_col).T -
            gamma,
            (self.data.num_rows * self.data.num_cols, 1),
            order='F'
        )

        reshaped_d_loss = np.reshape(d_square_loss, (self.data.num_rows, self.data.num_cols))

        grad_at_opt = reshaped_d_loss + self.fmodel.current_lambdas[0] * u_hat * np.sign(sigma_hat) * v_hat.T
        print "grad_at_opt wrt gamma", grad_at_opt
        assert(np.all(np.abs(grad_at_opt) < 1e-2))

    def _get_train_mask(self):
        train_mask = np.matrix(np.zeros((self.data.num_rows, self.data.num_cols))).T
        train_vec_mask = train_mask.flatten('F').A[0]
        print "train_vec_mask", self.data.train_idx
        train_vec_mask[self.data.train_idx] = 1
        assert(np.sum(train_vec_mask) == 2)
        print "daig train_vec_mask", np.diag(train_vec_mask)
        return np.diag(train_vec_mask)

    def _make_constraint_is_small(self, constraint):
        return constraint == 0 #abs(constraint) < 1e-16

    def _get_gradient_lambbda1(self, alpha, beta, lambdas):
        dU_dlambda = Variable(self.data.num_rows, self.data.num_cols)
        dVT_dlambda = Variable(self.data.num_rows, self.data.num_cols)
        dSigma_dlambda = Variable(self.data.num_rows)
        dalpha_dlambda = Variable(self.data.num_row_features, 1)
        dbeta_dlambda = Variable(self.data.num_col_features, 1)

        num_train = self.data.train_idx.size
        d_square_loss = 1.0/num_train * vec(
            self.data.row_features * dalpha_dlambda * self.onesT_row +
            (self.data.col_features * dbeta_dlambda * self.onesT_col).T
        )

        def _make_alpha_constraint(i):
            if np.abs(alpha[i]) > 1e-5:
                return self._make_constraint_is_small(
                    (self.train_vec_diag * d_square_loss).T * vec(self.data.row_features[:, i] * self.onesT_row)
                    + alpha[i] + lambdas[1] * dalpha_dlambda[i]
                )
            else:
                return dalpha_dlambda[i] == 0

        def _make_beta_constraint(i):
            if np.abs(beta[i]) > 1e-5:
                return self._make_constraint_is_small(
                    (self.train_vec_diag * d_square_loss).T * vec(self.data.col_features[:, i] * self.onesT_col)
                    + beta[i] + lambdas[1] * dbeta_dlambda[i]
                )
            else:
                return dbeta_dlambda[i] == 0

        constraints_dalpha = [
            _make_alpha_constraint(i) for i in range(self.data.num_row_features)
        ]
        constraints_dbeta = [
            _make_beta_constraint(i) for i in range(self.data.num_col_features)
        ]
        grad_problem = Problem(Minimize(0), constraints_dalpha + constraints_dbeta)
        grad_problem.solve()
        print grad_problem.status
        print "dalpha_dlambda1", dalpha_dlambda.value
        print "dbeta_dlambda1", dbeta_dlambda.value
        return {
            "dalpha_dlambda1": dalpha_dlambda,
            "dbeta_dlambda1": dbeta_dlambda,
            # "dgamma_dlambda0": dgamma_dlambda0,
        }

    def _get_threshold_sigma_diag(self, s, thres=1e-5):
        return np.diag((np.abs(s) > thres) * s)

    def _get_svd(self, gamma):
        # TODO: mask for nonzero parts
        u, s, v = np.linalg.svd(gamma)
        u_hat = u
        sigma_hat = self._get_threshold_sigma_diag(s)
        print "sigma_hat", sigma_hat
        v_hat = v.T
        return u_hat, sigma_hat, v_hat

    def _get_gradient_lambda0(self, alpha, beta, gamma, lambdas):
        dU_dlambda = Variable(self.data.num_rows, num_s_nonzero)
        dVT_dlambda = Variable(self.data.num_rows, num_s_nonzero)
        dSigma_dlambda = Variable(self.data.num_rows, 1)
        dSigma_dlambda = Variable(num_s_nonzero, 1)
        dalpha_dlambda = Variable(self.data.num_row_features, 1)
        dbeta_dlambda = Variable(self.data.num_col_features, 1)

        u_hat, sigma_hat, v_hat = self._get_svd(gamma)

        num_train = self.data.train_idx.size

        dsigma_contraints = []
        num_s_nonzero = sum([sigma_hat[i,i] == 0 for i in range(self.data.num_rows)])

        dgamma_dlambda = dU_dlambda * sigma_hat * v_hat.T + u_hat * diag(dSigma_dlambda) * v_hat.T + u_hat * sigma_hat * dVT_dlambda
        d_square_loss = 1.0/num_train * vec(
            self.data.row_features * dalpha_dlambda * self.onesT_row +
            (self.data.col_features * dbeta_dlambda * self.onesT_col).T +
            dgamma_dlambda
        )
        constraints_dgamma = (
            self._make_constraint_is_small(
                reshape(self.train_vec_diag * d_square_loss, self.data.num_rows, self.data.num_cols) +
                u_hat * np.sign(sigma_hat) * v_hat.T +
                lambdas[0] * dU_dlambda * np.sign(sigma_hat) * v_hat.T +
                lambdas[0] * u_hat * np.sign(sigma_hat) * dVT_dlambda
            )
        )

        def _make_alpha_constraint(i):
            if np.abs(alpha[i]) > 1e-5:
                return self._make_constraint_is_small(
                    (self.train_vec_diag * d_square_loss).T * vec(self.data.row_features[:, i] * self.onesT_row)
                    + lambdas[1] * dalpha_dlambda[i]
                )
            else:
                return dalpha_dlambda[i] == 0

        def _make_beta_constraint(i):
            if np.abs(beta[i]) > 1e-5:
                return self._make_constraint_is_small(
                    (self.train_vec_diag * d_square_loss).T * np.reshape(self.data.col_features[:, i] * self.onesT_col)
                    + lambdas[1] * dbeta_dlambda[i]
                )
            else:
                return dbeta_dlambda[i] == 0

        constraints_dalpha = [
            (
                _make_alpha_constraint(i)
                # (train_vec_diag * d_square_loss).T * np.reshape(self.data.row_features[:, i] * onesT_row, (self.data.num_rows * self.data.num_cols, 1), order="F")
                # + lambdas[1] * dalpha_dlambda[i] == 0
            )
            for i in range(self.data.num_row_features)
        ]
        constraints_dbeta = [
            (
                _make_beta_constraint(i)
                # (train_vec_diag * d_square_loss).T * np.reshape(self.data.col_features[:, i] * onesT_col, (self.data.num_rows * self.data.num_cols, 1), order="F")
                # + lambdas[1] * dbeta_dlambda[i] == 0
            )
            for i in range(self.data.num_col_features)
        ]

        constraints_uu_vv = [
            self._make_constraint_is_small(u_hat.T * dU_dlambda + dU_dlambda.T * u_hat),
            self._make_constraint_is_small(dVT_dlambda * v_hat + v_hat.T * dVT_dlambda.T),
        ]

        constraints = dsigma_contraints + constraints_dalpha + constraints_dbeta + [constraints_dgamma] + constraints_uu_vv

        grad_problem = Problem(Minimize(0), constraints)

        # grad_problem = Problem(Minimize(
        #     norm(u_hat.T * dU_dlambda + dU_dlambda.T * u_hat, 2) + norm(dVT_dlambda * v_hat + v_hat.T * dVT_dlambda.T, 2)
        # ), constraints)
        grad_problem.solve()

        print grad_problem.status
        print "dU_dlambda0", dU_dlambda.value
        print "dgamma_dlambda0", dgamma_dlambda.value
        print "dalpha_dlambda0", dalpha_dlambda.value
        print "dbeta_dlambda0", dbeta_dlambda.value
        return {
            "dalpha_dlambda0": dalpha_dlambda,
            "dbeta_dlambda0": dbeta_dlambda,
            "dgamma_dlambda0": dgamma_dlambda,
        }
