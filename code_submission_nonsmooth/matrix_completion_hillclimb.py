from cvxpy import *
import cvxopt
import numpy as np
import scipy as sp
from common import testerror_matrix_completion, get_matrix_completion_fitted_values
from common import make_column_major_flat, make_column_major_reshape
from gradient_descent_algo import Gradient_Descent_Algo
from convexopt_solvers import MatrixCompletionProblemWrapperSimple #, MatrixCompletionProblemWrapperStupid

class Lamdba_Deriv_Problem_Wrapper:
    def __init__(self, num_rows, alpha_size, beta_size):
        self.dU_dlambda = Variable(num_rows, num_rows)
        self.dV_dlambda = Variable(num_rows, num_rows)
        self.dSigma_dlambda = Variable(num_rows, 1)
        self.dalpha_dlambda = Variable(alpha_size, 1) if alpha_size > 0 else None
        self.dbeta_dlambda = Variable(beta_size, 1) if beta_size > 0 else None

class Matrix_Completion_Hillclimb_Base(Gradient_Descent_Algo):
    def _create_descent_settings(self):
        self.num_iters = 20
        self.step_size_init = 1
        self.step_size_min = 1e-6
        self.shrink_factor = 0.1
        self.decr_enough_threshold = 1e-4 * 5
        self.use_boundary = True
        self.boundary_factor = 0.999999
        self.backtrack_alpha = 0.001

        self.zero_thres = 1e-6 # determining which values are zero

        assert(self.data.num_rows == self.data.num_cols)

        self.train_vec_diag = self._get_mask(self.data.train_idx)
        self.val_vec_diag = self._get_mask(self.data.validate_idx)
        self.onesT_row = np.matrix(np.ones(self.data.num_rows))
        self.onesT_col = np.matrix(np.ones(self.data.num_cols))
        self.num_train = self.data.train_idx.size
        self.num_val = self.data.validate_idx.size

    def get_validate_cost(self, model_params):
        return testerror_matrix_completion(
            self.data,
            self.data.validate_idx,
            model_params
        )

    def _get_mask(self, idx):
        mask_shell = np.zeros(self.data.num_rows * self.data.num_cols)
        mask_shell[idx] = 1
        return np.diag(mask_shell)

    def _make_constraint_is_small(self, constraint):
        return constraint == 0 # abs(constraint) < 1e-16

    def _get_threshold_sigma_diag(self, s):
        return np.diag((np.abs(s) > self.zero_thres) * s)

    def _get_svd(self, gamma):
        u, s, v = np.linalg.svd(gamma)
        u_hat = u
        sigma_hat = self._get_threshold_sigma_diag(s)
        v_hat = v.T
        return u_hat, sigma_hat, v_hat

    def _get_svd_mini(self, gamma):
        u, s, v = np.linalg.svd(gamma)
        u_hat = u
        nonzero_mask = (np.abs(s) > self.zero_thres)
        u_mini = u[:, nonzero_mask]
        v_mini = v[nonzero_mask, :].T
        sigma_mini = np.diag(s[nonzero_mask])
        return u_mini, sigma_mini, v_mini

    def _get_nonzero_mini(self, alpha, beta):
        def _get_mini_mask(a):
            return np.where(np.abs(a) > self.zero_thres)[0]
        def _get_mini(a, mask):
            a = a[mask]
            return np.reshape(a, (a.size, 1))
        alpha_mask = _get_mini_mask(alpha)
        alpha_mini = _get_mini(alpha, alpha_mask)
        beta_mask = _get_mini_mask(beta)
        beta_mini = _get_mini(beta, beta_mask)
        mini_row_features = self.data.row_features[:, alpha_mask]
        mini_col_features = self.data.col_features[:, beta_mask]
        return alpha_mini, beta_mini, mini_row_features, mini_col_features

    def _get_val_gradient(self, grad_dict, alpha, beta, gamma, row_features, col_features):
        d_square_loss = self.data.observed_matrix - gamma
        model_grad = grad_dict["dgamma_dlambda"]
        if alpha.size > 0:
            d_square_loss -= row_features * alpha * self.onesT_row
            model_grad += row_features * grad_dict["dalpha_dlambda"] * self.onesT_row
        if beta.size > 0:
            d_square_loss -= (col_features * beta * self.onesT_col).T
            model_grad += (col_features * grad_dict["dbeta_dlambda"] * self.onesT_col).T

        dval_dlambda = - 1.0/self.num_val * (
            self.val_vec_diag
            * make_column_major_flat(d_square_loss)
        ).T * make_column_major_flat(model_grad)
        return dval_dlambda

    def print_model_details(self):
        alpha = self.fmodel.current_model_params["alpha"]
        beta = self.fmodel.current_model_params["beta"]
        gamma = self.fmodel.current_model_params["gamma"]
        u, s, v = self._get_svd_mini(gamma)
        self.log("alpha %s" % alpha)
        self.log("beta %s" % beta)
        self.log("sigma %s" % s)
        self.log("u %s" % u)
        self.log("v %s" % v)

        # check that the matrices are similar - sanity check
        self.log("data.real_matrix row 1 %s" % self.data.real_matrix[1,:])
        fitted_m = get_matrix_completion_fitted_values(
            self.data.row_features,
            self.data.col_features,
            self.fmodel.current_model_params["alpha"],
            self.fmodel.current_model_params["beta"],
            self.fmodel.current_model_params["gamma"]
        )
        self.log("fitted_m row 1 %s" % fitted_m[1,:])

    def _check_optimality_conditions(self, model_params, lambdas, opt_thres=1e-2):
        # lambdas must be an exploded lambda matrix
        assert(lambdas.size == 5)

        alpha = model_params["alpha"]
        beta = model_params["beta"]
        gamma = model_params["gamma"]

        u_hat, sigma_hat, v_hat = self._get_svd_mini(gamma)

        d_square_loss = -1.0/self.num_train * self.train_vec_diag * make_column_major_flat(
            self.data.observed_matrix
            - gamma
            - self.data.row_features * alpha * self.onesT_row
            - (self.data.col_features * beta * self.onesT_col).T
        )

        grad_at_opt_gamma = (
            u_hat.T * make_column_major_reshape(d_square_loss, (self.data.num_rows, self.data.num_cols)) * v_hat
            + lambdas[0] * np.sign(sigma_hat)
        )
        print "grad_at_opt wrt gamma (should be zero)", grad_at_opt_gamma

        grad_at_opt_alpha = []
        for i in range(alpha.size):
            alpha_sign = np.sign(alpha[i]) if np.abs(alpha[i]) > self.zero_thres else 0
            grad_at_opt_alpha.append((
                d_square_loss.T * make_column_major_flat(
                    self.data.row_features[:, i] * self.onesT_row
                )
                + lambdas[1] * alpha_sign
                + lambdas[2] * alpha[i]
            )[0,0])
        print "grad_at_opt wrt alpha (should be zero)", grad_at_opt_alpha

        grad_at_opt_beta = []
        for i in range(beta.size):
            beta_sign = np.sign(beta[i]) if np.abs(beta[i]) > self.zero_thres else 0
            grad_at_opt_beta.append((
                d_square_loss.T * make_column_major_flat(
                    (self.data.col_features[:, i] * self.onesT_col).T
                )
                + lambdas[3] * beta_sign
                + lambdas[4] * beta[i]
            )[0,0])
        print "grad_at_opt wrt beta (should be zero)", grad_at_opt_beta

        # not everything should be zero, if it is not differentiable at that point
        # assert(np.all(np.abs(grad_at_opt_gamma) < opt_thres))
        # assert(np.all(np.abs(grad_at_opt_alpha) < opt_thres))
        # assert(np.all(np.abs(grad_at_opt_beta) < opt_thres))

        return grad_at_opt_gamma, np.array(grad_at_opt_alpha), np.array(grad_at_opt_beta)

class Matrix_Completion_Hillclimb_Simple(Matrix_Completion_Hillclimb_Base):
    method_label = "Matrix_Completion_Hillclimb"

    def _create_lambda_configs(self):
        self.lambda_mins = [1e-6, 1e-6]

    def _create_problem_wrapper(self):
        self.problem_wrapper = MatrixCompletionProblemWrapperSimple(self.data)
        # self.problem_wrapper = MatrixCompletionProblemWrapperStupid(self.data)

    def _get_lambda_derivatives(self):
        alpha = self.fmodel.current_model_params["alpha"]
        beta = self.fmodel.current_model_params["beta"]
        gamma = self.fmodel.current_model_params["gamma"]

        u_hat, sigma_hat, v_hat = self._get_svd(gamma)

        alpha, beta, row_features, col_features = self._get_nonzero_mini(alpha, beta)

        print "alpha", alpha
        print "Beta", beta

        grad_dict0 = self._get_gradient_lambda0(
            alpha,
            beta,
            gamma,
            row_features,
            col_features,
            u_hat,
            sigma_hat,
            v_hat,
            self.fmodel.current_lambdas
        )
        dval_dlambda0 = self._get_val_gradient(grad_dict0, alpha, beta, gamma, row_features, col_features)
        print "dval_dlambda0", dval_dlambda0

        grad_dict1 = self._get_gradient_lambda1(
            alpha,
            beta,
            gamma,
            row_features,
            col_features,
            u_hat,
            sigma_hat,
            v_hat,
            self.fmodel.current_lambdas
        )
        dval_dlambda1 = self._get_val_gradient(grad_dict1, alpha, beta, gamma, row_features, col_features)

        print "FINAL dval_dlambda", np.array([dval_dlambda0, dval_dlambda1]).flatten()
        # assert(not dval_dlambda0 == 0)
        # assert(not dval_dlambda1 == 0)

        return np.array([dval_dlambda0, dval_dlambda1]).flatten()

    def _get_gradient_lambda0(
            self,
            alpha,
            beta,
            gamma,
            row_features,
            col_features,
            u_hat,
            sigma_hat,
            v_hat,
            lambdas,
        ):
        # we have a really complex set of linear equations. We will use cvxpy
        # to solve this
        dU_dlambda = Variable(self.data.num_rows, self.data.num_rows)
        dV_dlambda = Variable(self.data.num_rows, self.data.num_rows)
        dSigma_dlambda = Variable(self.data.num_rows, 1)
        dalpha_dlambda = Variable(alpha.size, 1) if alpha.size > 0 else None
        dbeta_dlambda = Variable(beta.size, 1) if beta.size > 0 else None

        # Constrain dsigma_dlambda to be zero for indices where sigma_i == 0
        constraints_sigma = []
        for i in range(sigma_hat.shape[0]):
            if sigma_hat[i,i] == 0:
                constraints_sigma.append(
                    dSigma_dlambda[i] == 0
                )

        d_square_loss = - 1.0/self.num_train * self.train_vec_diag * make_column_major_flat(
            self.data.observed_matrix
            - gamma
            - row_features * alpha * self.onesT_row
            - (col_features * beta * self.onesT_col).T
        )

        d_square_loss_reshape = make_column_major_reshape(d_square_loss, (self.data.num_rows, self.data.num_cols))

        dgamma_dlambda = (
            dU_dlambda * sigma_hat * v_hat.T
            + u_hat * diag(dSigma_dlambda) * v_hat.T
            + u_hat * sigma_hat * dV_dlambda.T
        )
        dd_square_loss = dgamma_dlambda
        if alpha.size > 0:
            dd_square_loss += row_features * dalpha_dlambda * self.onesT_row
        if beta.size > 0:
            dd_square_loss += (col_features * dbeta_dlambda * self.onesT_col).T
        dd_square_loss = 1.0/self.num_train * self.train_vec_diag * vec(dd_square_loss)

        dsigma_mask = np.ones(sigma_hat.shape)
        for i in range(sigma_hat.shape[0]):
            if sigma_hat[i,i] == 0:
                dsigma_mask[i,i] = 0
        dsigma_mask = make_column_major_flat(dsigma_mask)
        dsigma_mask = np.diag(dsigma_mask.flatten())

        # Constraint from implicit differentiation of the optimality conditions
        # that were defined by taking the gradient of the training objective wrt gamma
        constraints_dgamma = [
            dsigma_mask * vec(
                dU_dlambda.T * d_square_loss_reshape * v_hat
                + u_hat.T * reshape(
                    dd_square_loss,
                    self.data.num_rows,
                    self.data.num_cols,
                ) * v_hat
                + u_hat.T * d_square_loss_reshape * dV_dlambda
                + np.sign(sigma_hat)
            ) == np.zeros((self.data.num_rows * self.data.num_cols, 1))
        ]

        # Constraint from definition of U^T U = I and same for V
        constraints_uu_vv = [
            self._make_constraint_is_small(u_hat.T * dU_dlambda + dU_dlambda.T * u_hat),
            self._make_constraint_is_small(dV_dlambda.T * v_hat + v_hat.T * dV_dlambda),
        ]

        def _make_alpha_constraint(i):
            if np.abs(alpha[i]) > self.zero_thres:
                return self._make_constraint_is_small(
                    dd_square_loss.T * vec(row_features[:, i] * self.onesT_row)
                    + lambdas[1] * dalpha_dlambda[i]
                )
            else:
                return dalpha_dlambda[i] == 0

        def _make_beta_constraint(i):
            if np.abs(beta[i]) > self.zero_thres:
                return self._make_constraint_is_small(
                    dd_square_loss.T * vec(
                        (col_features[:, i] * self.onesT_col).T
                    )
                    + lambdas[1] * dbeta_dlambda[i]
                )
            else:
                return dbeta_dlambda[i] == 0

        # Constraint from implicit differentiation of the optimality conditions
        # that were defined by taking the gradient of the training objective wrt
        # alpha and beta, respectively
        constraints_dalpha = [_make_alpha_constraint(i) for i in range(alpha.size)]
        constraints_dbeta = [_make_beta_constraint(i) for i in range(beta.size)]

        constraints = constraints_sigma + constraints_dgamma + constraints_uu_vv + constraints_dalpha + constraints_dbeta
        grad_problem = Problem(Minimize(0), constraints)
        grad_problem.solve(solver=ECOS, max_iters=500)
        print "grad_problem.status", grad_problem.status
        print "dU_dlambda", dU_dlambda.value
        print "dV_dlambda", dV_dlambda.value
        print "dSigma_dlambda", dSigma_dlambda.value
        print "dgamma_dlambda0", dgamma_dlambda.value
        print "dalpha_dlambda0", dalpha_dlambda.value if dalpha_dlambda is not None else 0
        print "dbeta_dlambda0", dbeta_dlambda.value if dbeta_dlambda is not None else 0
        assert(grad_problem.status == OPTIMAL)
        return {
            "dalpha_dlambda": dalpha_dlambda.value if dalpha_dlambda is not None else 0,
            "dbeta_dlambda": dbeta_dlambda.value if dbeta_dlambda is not None else 0,
            "dgamma_dlambda": dgamma_dlambda.value if dSigma_dlambda is not None else 0,
        }

    def _get_gradient_lambda1(
            self,
            alpha,
            beta,
            gamma,
            row_features,
            col_features,
            u_hat,
            sigma_hat,
            v_hat,
            lambdas,
        ):
        dU_dlambda = Variable(self.data.num_rows, self.data.num_cols)
        dV_dlambda = Variable(self.data.num_rows, self.data.num_cols)
        dSigma_dlambda = Variable(self.data.num_rows)
        dalpha_dlambda = Variable(alpha.size, 1) if alpha.size > 0 else None
        dbeta_dlambda = Variable(beta.size, 1) if beta.size > 0 else None

        # Constrain dsigma_dlambda to be zero for indices where sigma_i == 0
        constraints_sigma = []
        for i in range(sigma_hat.shape[0]):
            if sigma_hat[i,i] == 0:
                constraints_sigma.append(
                    dSigma_dlambda[i] == 0
                )

        d_square_loss = - 1.0/self.num_train * self.train_vec_diag * make_column_major_flat(
            self.data.observed_matrix
            - gamma
            - row_features * alpha * self.onesT_row
            - (col_features * beta * self.onesT_col).T
        )

        d_square_loss_reshape = make_column_major_reshape(d_square_loss, (self.data.num_rows, self.data.num_cols))

        dgamma_dlambda = (
            dU_dlambda * sigma_hat * v_hat.T
            + u_hat * diag(dSigma_dlambda) * v_hat.T
            + u_hat * sigma_hat * dV_dlambda.T
        )
        dd_square_loss = dgamma_dlambda
        if alpha.size > 0:
            dd_square_loss += row_features * dalpha_dlambda * self.onesT_row
        if beta.size > 0:
            dd_square_loss += (col_features * dbeta_dlambda * self.onesT_col).T
        dd_square_loss = 1.0/self.num_train * self.train_vec_diag * vec(dd_square_loss)

        dsigma_mask = np.ones(sigma_hat.shape)
        for i in range(sigma_hat.shape[0]):
            if sigma_hat[i,i] == 0:
                dsigma_mask[i,i] = 0
        dsigma_mask = make_column_major_flat(dsigma_mask)
        dsigma_mask = np.diag(dsigma_mask.flatten())

        # Constraint from implicit differentiation of the optimality conditions
        # that were defined by taking the gradient of the training objective wrt gamma
        constraints_dgamma = [
            dsigma_mask * vec(
                dU_dlambda.T * d_square_loss_reshape * v_hat
                + u_hat.T * reshape(
                    dd_square_loss,
                    self.data.num_rows,
                    self.data.num_cols,
                ) * v_hat
                + u_hat.T * d_square_loss_reshape * dV_dlambda
            ) == np.zeros((self.data.num_rows * self.data.num_cols, 1))
        ]

        # Constraint from definition of U^T U = I and same for V
        constraints_uu_vv = [
            self._make_constraint_is_small(u_hat.T * dU_dlambda + dU_dlambda.T * u_hat),
            self._make_constraint_is_small(dV_dlambda.T * v_hat + v_hat.T * dV_dlambda),
        ]

        def _make_alpha_constraint(i):
            if np.abs(alpha[i]) > self.zero_thres:
                return self._make_constraint_is_small(
                    dd_square_loss.T * vec(row_features[:, i] * self.onesT_row)
                    + np.sign(alpha[i]) + alpha[i]
                    + lambdas[1] * dalpha_dlambda[i]
                )
            else:
                return dalpha_dlambda[i] == 0

        def _make_beta_constraint(i):
            if np.abs(beta[i]) > self.zero_thres:
                return self._make_constraint_is_small(
                    dd_square_loss.T * vec(
                        (col_features[:, i] * self.onesT_col).T
                    )
                    + np.sign(beta[i]) + beta[i]
                    + lambdas[1] * dbeta_dlambda[i]
                )
            else:
                return dbeta_dlambda[i] == 0

        # Constraint from implicit differentiation of the optimality conditions
        # that were defined by taking the gradient of the training objective wrt
        # alpha and beta, respectively
        constraints_dalpha = [_make_alpha_constraint(i) for i in range(alpha.size)]
        constraints_dbeta = [_make_beta_constraint(i) for i in range(beta.size)]

        constraints = constraints_sigma + constraints_dgamma + constraints_uu_vv + constraints_dalpha + constraints_dbeta
        grad_problem = Problem(Minimize(0), constraints)
        grad_problem.solve(solver=ECOS, max_iters=500)
        print "grad_problem.status 1", grad_problem.status
        print "dU_dlambda1", dU_dlambda.value
        print "dV_dlambda1", dV_dlambda.value
        print "dSigma_dlambda1", dSigma_dlambda.value
        print "dgamma_dlambda1", dgamma_dlambda.value
        print "dalpha_dlambda1", dalpha_dlambda.value if dalpha_dlambda is not None else 0
        print "dbeta_dlambda1", dbeta_dlambda.value if dbeta_dlambda is not None else 0
        assert(grad_problem.status in [OPTIMAL, OPTIMAL_INACCURATE])
        return {
            "dalpha_dlambda": dalpha_dlambda.value if dalpha_dlambda is not None else 0,
            "dbeta_dlambda": dbeta_dlambda.value if dbeta_dlambda is not None else 0,
            "dgamma_dlambda": dgamma_dlambda.value if dSigma_dlambda is not None else 0,
        }

    def _double_check_derivative_indepth(self, i, model1, model2, model0, eps):
        if i == 0:
            self._double_check_derivative_indepth_lambda0(model1, model2, model0, eps)
        return

    def _double_check_derivative_indepth_lambda0(self, model1, model2, model0, eps):
        # not everything should be zero, if it is not differentiable at that point
        dalpha_dlambda = (model1["alpha"] - model2["alpha"])/(eps * 2)
        dbeta_dlambda = (model1["beta"] - model2["beta"])/(eps * 2)

        gamma1 = model1["gamma"]
        u1, s1, v1 = self._get_svd_mini(gamma1)
        gamma2 = model2["gamma"]
        u2, s2, v2 = self._get_svd_mini(gamma2)
        gamma0 = model0["gamma"]
        u_hat, sigma_hat, v_hat = self._get_svd_mini(gamma0)
        dU_dlambda = (u1 - u2)/(eps * 2)
        dV_dlambda = (v1 - v2)/(eps*2)
        dSigma_dlambda = (s1 - s2)/(eps * 2)
        dgamma_dlambda = (gamma1 - gamma2)/(eps * 2)

        print "dalpha_dlambda0, %s" % (dalpha_dlambda)
        print "dBeta_dlambda0, %s" % (dbeta_dlambda)
        print "dU_dlambda0", dU_dlambda
        print "ds_dlambda0, %s" % (dSigma_dlambda)
        print "dgamma_dlambda0, %s" % (dgamma_dlambda)

        split_dgamma_dlambda = dU_dlambda * sigma_hat * v_hat.T + u_hat * dSigma_dlambda * v_hat.T + u_hat * sigma_hat * dV_dlambda.T

        # print "alpha1", model1["alpha"]
        # print 'alpha2', model2["alpha"]
        # print "eps", eps
        # print "u_hat", u_hat
        # print "u1", u1
        # print "u2", u2
        # print "v_hat", v_hat
        # print "v1", v1
        # print "v2", v2
        # print "sigma_hat", sigma_hat
        # print "s1", s1
        # print "s2", s2

        print "should be zero? dU_dlambda * u.T", u_hat.T * dU_dlambda + dU_dlambda.T * u_hat
        print "should be zero? dv_dlambda * v.T", dV_dlambda.T * v_hat + v_hat.T * dV_dlambda

        print "should be zero? dgamma_dlambda - dgamma_dlambda", split_dgamma_dlambda - dgamma_dlambda

        d_square_loss = 1.0/self.num_train * self.train_vec_diag * make_column_major_flat(
            dgamma_dlambda
            + self.data.row_features * dalpha_dlambda * self.onesT_row
            + (self.data.col_features * dbeta_dlambda * self.onesT_col).T
        )

        dalpha_dlambda_imp = []
        for i in range(dalpha_dlambda.size):
            dalpha_dlambda_imp.append((
                d_square_loss.T * make_column_major_flat(self.data.row_features[:, i] * self.onesT_row)
                + self.fmodel.current_lambdas[1] * dalpha_dlambda[i]
            )[0,0])
        print "should be zero? numerical plugin to the imp deriv eqn, dalpha_dlambda", dalpha_dlambda_imp

        db_dlambda_imp = []
        for i in range(dbeta_dlambda.size):
            db_dlambda_imp.append((
                d_square_loss.T * make_column_major_flat(
                    (self.data.col_features[:, i] * self.onesT_col).T
                )
                + self.fmodel.current_lambdas[1] * dbeta_dlambda[i]
            )[0,0])
        print "should be zero? numerical plugin to the imp deriv eqn, dbeta_dlambda_imp", db_dlambda_imp

        print "should be zero? numerical plugin to the imp deriv eqn, dgamma_dlambda", (
            u_hat.T * make_column_major_reshape(d_square_loss, (self.data.num_rows, self.data.num_cols)) * v_hat +
                np.sign(sigma_hat)
                + u_hat.T *  self.fmodel.current_lambdas[0] * dU_dlambda * np.sign(sigma_hat)
                + self.fmodel.current_lambdas[0] * np.sign(sigma_hat) * dV_dlambda.T * v_hat
        )

        # print "==== numerical numerical implicit derivatives===="
        # print "eps", eps
        # dg1, da1, db1 = self._check_optimality_conditions(model1)
        # dg2, da2, db2 = self._check_optimality_conditions(model2)
        # print "should be zero - numerical deriv - dg", (dg1 - dg2)/(2 * eps)
        # print "should be zero - numerical deriv - da", (da1 - da2)/(2 * eps)
        # print "should be zero - numerical deriv - db", (db1 - db2)/(2 * eps)
