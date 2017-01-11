import time
import sys
import numpy as np
import scipy as sp
from sklearn.utils.extmath import randomized_svd
from common import make_column_major_flat, make_column_major_reshape, print_time
from common import get_matrix_completion_fitted_values, get_norm2

class MatrixCompletionProblem:
    NUM_LAMBDAS = 5
    step_size = 1.0
    min_step_size = 1e-6
    step_size_shrink = 0.75
    step_size_shrink_small = 0.2
    print_iter = 10000

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
        self.gamma_num_s = None
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

    def get_value(self, alpha, beta, gamma, given_nuc_norm=None):
        matrix_eval = make_column_major_flat(
            self.observed_matrix
            - get_matrix_completion_fitted_values(
                self.row_features,
                self.col_features,
                alpha,
                beta,
                gamma,
            )
        )
        square_loss = 0.5/self.num_train * get_norm2(
            matrix_eval[self.train_idx],
            power=2,
        )
        if given_nuc_norm is None:
            nuc_norm = self.lambdas[0] * np.linalg.norm(gamma, ord="nuc")
        else:
            nuc_norm = self.lambdas[0] * given_nuc_norm
        alpha_norm1 = self.lambdas[1] * np.linalg.norm(alpha, ord=1)
        alpha_norm2 = 0.5 * self.lambdas[2] * get_norm2(alpha, power=2)
        beta_norm1 = self.lambdas[3] * np.linalg.norm(beta, ord=1)
        beta_norm2 = 0.5 * self.lambdas[4] * get_norm2(beta, power=2)
        return square_loss + nuc_norm + alpha_norm1 + alpha_norm2 + beta_norm1 + beta_norm2

    def update(self, lambdas):
        assert(lambdas.size == self.lambdas.size)
        self.lambdas = lambdas

    def solve(self, max_iters=1000, tol=1e-5):
        start_time = time.time()
        step_size = self.step_size
        old_val = self.get_value(
            self.alpha_curr,
            self.beta_curr,
            self.gamma_curr
        )
        val_drop = 0
        for i in range(max_iters):
            if i % self.print_iter == 0:
                print "iter %d: cost %f time %f (step size %f)" % (i, old_val, time.time() - start_time, step_size)
                print "num zeros", self.gamma_num_s
                sys.stdout.flush()

            gamma_grad, alpha_grad, beta_grad = self.get_smooth_gradient()
            potential_val, potential_alpha, potential_beta, potential_gamma = self.get_potential_values(
                step_size,
                alpha_grad,
                beta_grad,
                gamma_grad,
            )
            while old_val < potential_val and step_size > self.min_step_size:
                print "potential val is bigger: %f, %f" % (old_val, potential_val)
                if potential_val > 10 * old_val:
                    step_size *= self.step_size_shrink_small
                else:
                    step_size *= self.step_size_shrink
                potential_val, potential_alpha, potential_beta, potential_gamma = self.get_potential_values(
                    step_size,
                    alpha_grad,
                    beta_grad,
                    gamma_grad,
                )

            val_drop = old_val - potential_val
            if old_val >= potential_val:
                self.alpha_curr = potential_alpha
                self.beta_curr = potential_beta
                self.gamma_curr = potential_gamma
                old_val = potential_val
                if val_drop < tol:
                    print "decrease is too small"
                    break
            else:
                print "step size too small. increased in cost"
                break
        val_drop_str = "(log10) %s" % np.log10(val_drop) if val_drop > 0 else val_drop
        print "fin cost %f, solver diff %s, steps %d" % (old_val, val_drop_str, i)
        print "tot time", time.time() - start_time
        return self.gamma_curr, self.alpha_curr, self.beta_curr

    def get_potential_values(self, step_size, alpha_grad, beta_grad, gamma_grad):
        try:
            potential_gamma, nuc_norm = self.get_prox_nuclear(
                self.gamma_curr - step_size * gamma_grad,
                step_size * self.lambdas[0]
            )
        except np.linalg.LinAlgError:
            print "SVD did not converge - ignore proximal gradient step for nuclear norm"
            potential_gamma = self.gamma_curr - step_size * gamma_grad
            nuc_norm = None

        potential_alpha = self.get_prox_l1(
            self.alpha_curr - step_size * alpha_grad,
            step_size * self.lambdas[1]
        )
        potential_beta = self.get_prox_l1(
            self.beta_curr - step_size * beta_grad,
            step_size * self.lambdas[3]
        )
        potential_val = self.get_value(potential_alpha, potential_beta, potential_gamma, given_nuc_norm=nuc_norm)
        return potential_val, potential_alpha, potential_beta, potential_gamma

    # @print_time
    def get_smooth_gradient(self):
        # @print_time
        def _get_dsquare_loss():
            d_square_loss = - 1.0/self.num_train * make_column_major_flat(
                self.observed_matrix
                - get_matrix_completion_fitted_values(
                    self.row_features,
                    self.col_features,
                    self.alpha_curr,
                    self.beta_curr,
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
        def _get_alpha_grad():
            row_features_one = np.tile(self.row_features, (self.num_rows, 1))
            alpha_grad = (d_square_loss.T *  row_features_one).T + self.lambdas[2] * self.alpha_curr
            return alpha_grad

        # @print_time
        def _get_beta_grad():
            col_features_one = np.repeat(self.col_features, [self.num_rows] * self.row_features.shape[0], axis=0)
            beta_grad = (d_square_loss.T *  col_features_one).T + self.lambdas[4] * self.beta_curr
            return beta_grad

        return _get_gamma_grad(), _get_alpha_grad(), _get_beta_grad()

    # @print_time
    def get_prox_nuclear(self, x_matrix, scale_factor):
        # prox of function scale_factor * nuclear_norm
        # 15 is a random threshold to decide which solver to use.
        if self.gamma_num_s is None or self.gamma_num_s > 15:
            u, s, vt = np.linalg.svd(x_matrix, full_matrices=False)
        else:
            # This is a bit faster for bigger matrices
            u, s, vt = randomized_svd(
                x_matrix,
                n_components=self.gamma_num_s,
                n_iter=1,
                random_state=None,
            )
            u = np.matrix(u)
            vt = np.matrix(vt)

        num_nonzero_orig = (np.where(s > scale_factor))[0].size
        thres_s = np.maximum(s - scale_factor, 0)
        nuc_norm = np.linalg.norm(thres_s, ord=1)
        self.gamma_num_s = (np.where(thres_s > 0))[0].size
        return u * np.diag(thres_s) * vt, nuc_norm

    # @print_time
    def get_prox_l1(self, x_vector, scale_factor):
        thres_x = np.maximum(x_vector - scale_factor, 0) - np.maximum(-x_vector - scale_factor, 0)
        return thres_x
