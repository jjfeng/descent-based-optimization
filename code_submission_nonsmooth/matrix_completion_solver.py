import time
import sys
import numpy as np
from common import make_column_major_flat, make_column_major_reshape

class MatrixCompletionProblem:
    NUM_LAMBDAS = 5
    step_size = 0.8
    step_size_shrink = 0.75

    def __init__(self, data):
        self.num_rows = data.num_rows
        self.num_cols = data.num_cols
        self.num_row_features = data.num_row_features
        self.num_col_features = data.num_col_features
        self.num_train = data.train_idx.size
        self.train_idx = data.train_idx
        self.non_train_mask_vec = self._get_nontrain_mask(data.train_idx)
        self.observed_matrix = self._get_masked(data.observed_matrix)
        self.row_features = data.row_features
        self.col_features = data.col_features

        self.onesT_row = np.matrix(np.ones(self.num_rows))
        self.onesT_col = np.matrix(np.ones(self.num_cols))

        self.gamma_curr = np.zeros((data.num_rows, data.num_cols))
        self.alpha_curr = np.zeros((data.num_row_features,1))
        self.beta_curr = np.zeros((data.num_col_features,1))

        # just random setting to lambdas. this should be overriden
        self.lambdas = np.ones((self.NUM_LAMBDAS, 1))

    def _get_nontrain_mask(self, train_idx):
        non_train_mask_vec = np.ones(self.num_rows * self.num_cols, dtype=bool)
        non_train_mask_vec[train_idx] = False
        return non_train_mask_vec

    def _get_masked(self, obs_matrix):
        masked_obs_vec = make_column_major_flat(obs_matrix)
        masked_obs_vec[self.non_train_mask_vec] = 0
        masked_obs_m = make_column_major_reshape(
            masked_obs_vec,
            (self.num_rows, self.num_cols)
        )
        return masked_obs_m

    def get_value(self):
        matrix_eval = make_column_major_flat(
            self.observed_matrix
            - self.gamma_curr
            - self.row_features * self.alpha_curr * self.onesT_row
            - (self.col_features * self.beta_curr * self.onesT_col).T
        )
        square_loss = 0.5/self.num_train * np.power(np.linalg.norm(
            matrix_eval[self.train_idx],
            ord=None
        ), 2)
        nuc_norm = self.lambdas[0] * np.linalg.norm(self.gamma_curr, ord="nuc")
        alpha_norm1 = self.lambdas[1] * np.linalg.norm(self.alpha_curr, ord=1)
        alpha_norm2 = 0.5 * self.lambdas[2] * np.power(np.linalg.norm(self.alpha_curr, ord=None), 2)
        beta_norm1 = self.lambdas[3] * np.linalg.norm(self.beta_curr, ord=1)
        beta_norm2 = 0.5 * self.lambdas[4] * np.power(np.linalg.norm(self.beta_curr, ord=None), 2)
        return square_loss + nuc_norm + alpha_norm1 + alpha_norm2 + beta_norm1 + beta_norm2

    def update(self, lambdas):
        assert(lambdas.size == self.lambdas.size)
        self.lambdas = lambdas

    def solve(self, max_iters=1000, tol=1e-5):
        step_size = self.step_size
        old_val = None
        for i in range(max_iters):
            # print "self.gamma_curr", self.gamma_curr
            # print "self.alpha_curr", self.alpha_curr
            # print "self.beta_curr", self.beta_curr
            if i % 1000 == 0:
                print "iter %d: cost %f (step size %f)" % (i, self.get_value(), step_size)
                sys.stdout.flush()

            # if old_val is not None:
            #     assert(old_val >= self.get_value() - 1e-10)
            old_val = self.get_value()
            gamma_grad, alpha_grad, beta_grad = self.get_smooth_gradient()
            # print "gamma_grad, alpha_grad, beta_grad", gamma_grad
            # print alpha_grad
            # print beta_grad
            try:
                self.gamma_curr, num_nonzero_sv = self.get_prox_nuclear(
                    self.gamma_curr - step_size * gamma_grad,
                    step_size * self.lambdas[0]
                )
                if i % 1000 == 0:
                    print "iter %d: num_nonzero_sv %d" % (i, num_nonzero_sv)
            except np.linalg.LinAlgError:
                print "SVD did not converge - ignore proximal gradient step for nuclear norm"
                self.gamma_curr = self.gamma_curr - step_size * gamma_grad

            self.alpha_curr = self.get_prox_l1(
                self.alpha_curr - step_size * alpha_grad,
                step_size * self.lambdas[1]
            )
            self.beta_curr = self.get_prox_l1(
                self.beta_curr - step_size * beta_grad,
                step_size * self.lambdas[3]
            )
            # print "old_val - self.get_value()", old_val - self.get_value()
            if old_val < self.get_value():
                print "curr val is bigger: %f, %f" % (old_val, self.get_value())
                step_size *= self.step_size_shrink
            elif old_val - self.get_value() < tol:
                print "diff is very small %f" % (old_val - self.get_value())
                break
        return self.gamma_curr, self.alpha_curr, self.beta_curr

    # @print_time
    def get_smooth_gradient(self):
        d_square_loss = - 1.0/self.num_train * make_column_major_flat(
            self.observed_matrix
            - self.gamma_curr
            - self.row_features * self.alpha_curr * self.onesT_row
            - (self.col_features * self.beta_curr * self.onesT_col).T
        )
        d_square_loss[self.non_train_mask_vec] = 0

        def _get_gamma_grad():
            return make_column_major_reshape(
                d_square_loss,
                (self.num_rows, self.num_cols)
            )

        def _get_alpha_grad():
            alpha_grad = []
            for i in range(self.num_row_features):
                alpha_i_grad = d_square_loss.T * make_column_major_flat(
                    self.row_features[:,i] * self.onesT_row
                ) + self.lambdas[2] * self.alpha_curr[i]
                alpha_grad.append(alpha_i_grad[0,0])
            return np.matrix(alpha_grad).T

        def _get_beta_grad():
            beta_grad = []
            for i in range(self.num_col_features):
                beta_i_grad = d_square_loss.T * make_column_major_flat(
                    (self.col_features[:,i] * self.onesT_col).T
                ) + self.lambdas[4] * self.beta_curr[i]
                beta_grad.append(beta_i_grad[0,0])
            return np.matrix(beta_grad).T

        return _get_gamma_grad(), _get_alpha_grad(), _get_beta_grad()

    # @print_time
    def get_prox_nuclear(self, x_matrix, scale_factor):
        # prox of function scale_factor * nuclear_norm
        u, s, vt = np.linalg.svd(x_matrix, full_matrices=False)
        thres_s = np.maximum(s - scale_factor, 0)
        num_nonzero = (np.where(thres_s > 0))[0].size
        return u * np.diag(thres_s) * vt, num_nonzero

    # @print_time
    def get_prox_l1(self, x_vector, scale_factor):
        thres_x = np.maximum(x_vector - scale_factor, 0) - np.maximum(-x_vector - scale_factor, 0)
        return thres_x
