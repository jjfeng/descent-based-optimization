import time
import sys
import numpy as np
import scipy as sp
from common import make_column_major_flat, make_column_major_reshape, print_time
from common import get_matrix_completion_groups_fitted_values

class MatrixCompletionGroupsProblem:
    step_size = 0.8
    step_size_shrink = 0.75
    step_size_shrink_small = 0.1
    print_iter = 1

    def __init__(self, data):
        self.num_rows = data.num_rows
        self.num_cols = data.num_cols

        self.num_row_features = data.row_features[0].shape[1]
        self.num_col_features = data.col_features[0].shape[1]
        self.num_row_groups = len(data.row_features)
        self.num_col_groups = len(data.col_features)

        self.num_train = data.train_idx.size
        self.train_idx = data.train_idx
        self.non_train_mask_vec = self._get_nontrain_mask(data.train_idx)
        self.observed_matrix = self._get_masked(data.observed_matrix)
        self.row_features = data.row_features
        self.col_features = data.col_features

        self.onesT_row = np.matrix(np.ones(self.num_rows))
        self.onesT_col = np.matrix(np.ones(self.num_cols))

        self.gamma_curr = np.zeros((data.num_rows, data.num_cols))
        self.alphas_curr = [np.matrix(np.zeros(self.num_row_features)).T] * self.num_row_groups
        self.betas_curr = [np.matrix(np.zeros(self.num_col_features)).T] * self.num_col_groups

        self.num_lambdas = num_row_groups + num_col_groups + 1

        # just random setting to lambdas. this should be overriden
        self.lambdas = np.ones((self.num_lambdas, 1))

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
            - get_matrix_completion_groups_fitted_values(
                self.row_features,
                self.col_features,
                self.alphas_curr,
                self.betas_curr,
                self.gamma_curr,
            )
        )
        square_loss = 0.5/self.num_train * np.power(np.linalg.norm(
            matrix_eval[self.train_idx],
            ord=None
        ), 2)
        nuc_norm = self.lambdas[0] * np.linalg.norm(self.gamma_curr, ord="nuc")
        alpha_pen = 0
        for i, a in enumerate(self.alphas_curr):
            # group lasso penalties
            alpha_pen += self.lambdas[1 + i] * np.linalg.norm(a, ord=2)
        beta_pen = 0
        for i, b in enumerate(self.betas_curr):
            # group lasso penalties
            beta_pen += self.lambdas[1 + self.num_row_groups + i] * np.linalg.norm(b, ord=2)
        return square_loss + nuc_norm + alpha_pen + beta_pen

    def update(self, lambdas):
        print "lambdas", lambdas
        print "self.lambdas", self.lambdas
        assert(lambdas.size == self.lambdas.size)
        self.lambdas = lambdas

    def solve(self, max_iters=1000, tol=1e-5):
        start_time = time.time()
        step_size = self.step_size
        old_val = None
        for i in range(max_iters):
            # print "self.gamma_curr", self.gamma_curr
            # print "self.alpha_curr", self.alpha_curr
            # print "self.beta_curr", self.beta_curr
            if i % self.print_iter == 0:
                print "iter %d: cost %f time %f (step size %f)" % (i, self.get_value(), time.time() - start_time, step_size)
                sys.stdout.flush()

            if i == 5:
                1/0

            # if old_val is not None:
            #     assert(old_val >= self.get_value() - 1e-10)
            old_val = self.get_value()
            gamma_grad, alphas_grad, betas_grad = self.get_smooth_gradient()
            self.alphas_curr = [a - step_size * g for a, g in zip(self.alphas_curr, alphas_grad)]
            self.betas_curr = [b - step_size * g for b, g in zip(self.betas_curr, betas_grad)]
            try:
                self.gamma_curr, num_nonzero_sv = self.get_prox_nuclear(
                    self.gamma_curr - step_size * gamma_grad,
                    step_size * self.lambdas[0]
                )
                if i % self.print_iter == 0:
                    print "iter %d: num_nonzero_sv %d" % (i, num_nonzero_sv)
            except np.linalg.LinAlgError:
                print "SVD did not converge - ignore proximal gradient step for nuclear norm"
                self.gamma_curr = self.gamma_curr - step_size * gamma_grad

            for i, a_g_tuple in enumerate(zip(self.alphas_curr, alphas_grad)):
                a, g = a_g_tuple
                self.alphas_curr[i] = self.get_prox_l2(
                    a - step_size * g,
                    step_size * self.lambdas[1 + i]
                )
            for i, b_g_tuple in enumerate(zip(self.betas_curr, betas_grad)):
                self.betas_curr[i] = self.get_prox_l2(
                    b - step_size * g,
                    step_size * self.lambdas[1 + self.num_row_groups + i]
                )
            # print "old_val - self.get_value()", old_val - self.get_value()
            if old_val < self.get_value():
                print "curr val is bigger: %f, %f" % (old_val, self.get_value())
                step_size *= self.step_size_shrink
                if self.get_value() > 10 * old_val:
                    step_size *= self.step_size_shrink_small
            elif old_val - self.get_value() < tol:
                break
        print "solver diff (log10) %f" % np.log10(old_val - self.get_value())
        return self.gamma_curr, self.alphas_curr, self.betas_curr

    # @print_time
    def get_smooth_gradient(self):
        # @print_time
        def _get_dsquare_loss():
            d_square_loss = - 1.0/self.num_train * make_column_major_flat(
                self.observed_matrix
                - get_matrix_completion_groups_fitted_values(
                    self.row_features,
                    self.col_features,
                    self.alphas_curr,
                    self.betas_curr,
                    self.gamma_curr,
                )
            )
            d_square_loss[self.non_train_mask_vec] = 0
            return d_square_loss

        d_square_loss = _get_dsquare_loss()

        # @print_time
        def _get_gamma_grad():
            return make_column_major_reshape(
                d_square_loss,
                (self.num_rows, self.num_cols)
            )
        # @print_time
        def _get_alpha_grad(row_f):
            row_features_one = np.tile(row_f, (self.num_rows, 1))
            alpha_grad = (d_square_loss.T *  row_features_one).T
            return alpha_grad

        # @print_time
        def _get_beta_grad(col_f):
            col_features_one = np.repeat(col_f, [self.num_rows] * self.num_cols, axis=0)
            beta_grad = (d_square_loss.T *  col_features_one).T
            return beta_grad

        return _get_gamma_grad(), [_get_alpha_grad(f) for f in self.row_features], [_get_beta_grad(f) for f in self.row_features]

    # @print_time
    # This is the time sink! Can we make this faster?
    def get_prox_nuclear(self, x_matrix, scale_factor):
        # prox of function scale_factor * nuclear_norm
        u, s, vt = np.linalg.svd(x_matrix, full_matrices=False)
        thres_s = np.maximum(s - scale_factor, 0)
        num_nonzero = (np.where(thres_s > 0))[0].size
        return u * np.diag(thres_s) * vt, num_nonzero

    # @print_time
    def get_prox_l2(self, x_vector, scale_factor):
        thres_x = np.max(1 - scale_factor/np.linalg.norm(x_vector, ord=None), 0) * x_vector
        return thres_x
