import time
import sys
import numpy as np
from common import make_column_major_flat, make_column_major_reshape

class MatrixCompletionProblem:
    NUM_LAMBDAS = 5
    step_size = 0.1

    def __init__(self, data):
        self.num_rows = data.num_rows
        self.num_cols = data.num_cols
        self.num_row_features = data.num_row_features
        self.num_col_features = data.num_col_features
        self.num_train = data.train_idx.size
        self.train_mask = self._get_train_mask(data.train_idx)
        self.observed_matrix = self._get_masked(data.observed_matrix)
        self.row_features = data.row_features
        self.col_features = data.col_features

        self.onesT_row = np.matrix(np.ones(self.num_rows))
        self.onesT_col = np.matrix(np.ones(self.num_cols))

        self.gamma_curr = np.zeros((data.num_rows, data.num_cols))
        self.gamma_curr[0,0] = 1
        self.gamma_curr[1,1] = 0.2
        self.alpha_curr = np.zeros((data.num_row_features,1))
        self.beta_curr = np.zeros((data.num_col_features,1))
        self.lambdas = np.ones((self.NUM_LAMBDAS, 1))

    def _get_train_mask(self, train_idx):
        train_vec_mask = np.zeros(self.num_rows * self.num_cols)
        train_vec_mask[train_idx] = 1
        return np.diag(train_vec_mask)

    def _get_masked(self, obs_matrix):
        masked_obs_m = make_column_major_reshape(
            self.train_mask * make_column_major_flat(obs_matrix),
            (self.num_rows, self.num_cols)
        )
        return masked_obs_m

    def get_value(self):
        square_loss = 0.5/self.num_train * np.power(np.linalg.norm(
            self.train_mask * make_column_major_flat(
                self.observed_matrix
                - self.gamma_curr
                - self.row_features * self.alpha_curr * self.onesT_row
                - (self.col_features * self.beta_curr * self.onesT_col).T
            ),
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

    def solve(self, max_iters=100000, tol=1e-5):
        step_size = self.step_size
        old_val = None
        for i in range(max_iters):
            # print "self.gamma_curr", self.gamma_curr
            # print "self.alpha_curr", self.alpha_curr
            # print "self.beta_curr", self.beta_curr
            if i % 500 == 0:
                print "iter %d: cost %f" % (i, self.get_value())
                sys.stdout.flush()

            if old_val is not None:
                assert(old_val >= self.get_value() - 1e-10)
            old_val = self.get_value()
            gamma_grad, alpha_grad, beta_grad = self.get_smooth_gradient()
            # print "gamma_grad, alpha_grad, beta_grad", gamma_grad
            # print alpha_grad
            # print beta_grad
            try:
                self.gamma_curr = self.get_prox_nuclear(
                    self.gamma_curr - step_size * gamma_grad,
                    step_size * self.lambdas[0]
                )
            except LinAlgError:
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
            if old_val - self.get_value() < tol:
                print "diff is very small %f" % (old_val - self.get_value())
                break
        return self.gamma_curr, self.alpha_curr, self.beta_curr

    def get_smooth_gradient(self):
        d_square_loss = - 1.0/self.num_train * self.train_mask * make_column_major_flat(
            self.observed_matrix
            - self.gamma_curr
            - self.row_features * self.alpha_curr * self.onesT_row
            - (self.col_features * self.beta_curr * self.onesT_col).T
        )

        gamma_grad = make_column_major_reshape(
            d_square_loss,
            (self.num_rows, self.num_cols)
        )

        alpha_grad = []
        for i in range(self.num_row_features):
            alpha_i_grad = d_square_loss.T * make_column_major_flat(
                self.row_features[:,i] * self.onesT_row
            ) + self.lambdas[2] * self.alpha_curr[i]
            alpha_grad.append(alpha_i_grad[0,0])
        alpha_grad = np.matrix(alpha_grad).T

        beta_grad = []
        for i in range(self.num_col_features):
            beta_i_grad = d_square_loss.T * make_column_major_flat(
                (self.col_features[:,i] * self.onesT_col).T
            ) + self.lambdas[4] * self.beta_curr[i]
            beta_grad.append(beta_i_grad[0,0])
        beta_grad = np.matrix(beta_grad).T

        return gamma_grad, alpha_grad, beta_grad

    def get_prox_nuclear(self, x_matrix, scale_factor):
        # prox of function scale_factor * nuclear_norm
        u, s, vt = np.linalg.svd(x_matrix, full_matrices=False)
        thres_s = np.maximum(s - scale_factor, 0)
        return u * np.diag(thres_s) * vt

    def get_prox_l1(self, x_vector, scale_factor):
        thres_x = np.maximum(x_vector - scale_factor, 0) - np.maximum(-x_vector - scale_factor, 0)
        return thres_x
