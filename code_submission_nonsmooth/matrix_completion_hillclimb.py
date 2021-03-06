import time
from cvxpy import *
import cvxopt
import numpy as np
import scipy as sp
from common import VERBOSE
from common import print_time
from common import get_matrix_completion_fitted_values
from common import testerror_matrix_completion, get_matrix_completion_fitted_values
from common import make_column_major_flat, make_column_major_reshape
from gradient_descent_algo import Gradient_Descent_Algo
from convexopt_solvers import MatrixCompletionProblemWrapperSimple, MatrixCompletionProblemWrapperCustom

class Lamdba_Deriv_Problem_Wrapper:
    # A problem wrapper for solving for implicit derivatives.
    # The system of linear equations are quite complicated.
    # We will use cvxpy to solve them.
    max_iters = 100000
    eps = 1e-6
    solver=SCS
    acceptable_status = [OPTIMAL, OPTIMAL_INACCURATE]

    # @print_time
    def __init__(self, alpha, beta, u_hat, sigma_hat, v_hat):
        # you should send in minified versions of the SVD decomposition.
        # we are not interested in the eigenvectors for sigma = 0
        self.constraints_uu_vv = []
        self.dgamma_dlambda = np.zeros((u_hat.shape[0], v_hat.shape[0]))
        self.obj = 0

        self.dSigma_dlambda = None
        if sigma_hat.size > 0:
            self.dU_dlambda = Variable(u_hat.shape[0], u_hat.shape[1])
            self.dV_dlambda = Variable(v_hat.shape[0], v_hat.shape[1])
            self.dSigma_dlambda = Variable(sigma_hat.shape[0], 1)
            # Constraint from definition of U^T U = I and same for V
            uu = u_hat.T * self.dU_dlambda + self.dU_dlambda.T * u_hat
            vv = self.dV_dlambda.T * v_hat + v_hat.T * self.dV_dlambda
            self.constraints_uu_vv = [
                uu == 0,
                vv == 0,
            ]
            self.dgamma_dlambda = (
                self.dU_dlambda * sigma_hat * v_hat.T
                + u_hat * diag(self.dSigma_dlambda) * v_hat.T
                + u_hat * sigma_hat * self.dV_dlambda.T
            )

            self.obj += sum_squares(uu) + sum_squares(vv)

        self.dalpha_dlambda = Variable(alpha.size, 1) if alpha.size > 0 else None
        self.dbeta_dlambda = Variable(beta.size, 1) if beta.size > 0 else None

    @print_time
    # @param obj: backup is to minimize this objective function
    def solve(self, constraints, obj=0, big_thres=0.01):
        # The problem with solving the constrained problem is that it might be infeasible.
        # hence we want some things that were originally in the constraints to be in the objective
        # Don't use ECOS since it's very confused
        if self.dalpha_dlambda is not None or self.dbeta_dlambda is not None or self.dSigma_dlambda is not None:
            grad_problem = Problem(Minimize(self.obj + obj))
            grad_problem.solve(
                solver=self.solver,
                eps=self.eps,
                max_iters=self.max_iters,
                verbose=VERBOSE,
            )
            print "grad_problem.status", grad_problem.status, "value", grad_problem.value

            if grad_problem.value > big_thres:
                grad_problem.solve(
                    solver=self.solver,
                    eps=self.eps,
                    max_iters=self.max_iters * 2,
                    verbose=VERBOSE,
                    warm_start=True,
                )
                print "grad_problem.status", grad_problem.status, "value", grad_problem.value
                assert(grad_problem.status in self.acceptable_status)

        return {
            "dalpha_dlambda": self.dalpha_dlambda.value if self.dalpha_dlambda is not None else 0,
            "dbeta_dlambda": self.dbeta_dlambda.value if self.dbeta_dlambda is not None else 0,
            "dgamma_dlambda": self.dgamma_dlambda.value if self.dSigma_dlambda is not None else self.dgamma_dlambda,
            "dU_dlambda": self.dU_dlambda.value if self.dSigma_dlambda is not None else 0,
            "dV_dlambda": self.dV_dlambda.value if self.dSigma_dlambda is not None else 0,
            "dSigma_dlambda": self.dSigma_dlambda.value if self.dSigma_dlambda is not None else 0,
        }

class Matrix_Completion_Hillclimb_Base(Gradient_Descent_Algo):
    def _create_descent_settings(self):
        self.num_iters = 20
        self.step_size_init = 1
        self.step_size_min = 1e-6
        self.shrink_factor = 0.1
        self.decr_enough_threshold = 1e-4 * 5
        self.use_boundary = True
        self.boundary_factor = 0.8
        self.backtrack_alpha = 0.001

        self.zero_thres = 1e-6 # determining which values are zero

        assert(self.data.num_rows == self.data.num_cols)

        self.train_vec = self._get_vec_mask(self.data.train_idx)
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

    def _print_model_details(self):
        # overriding the function in Gradient_Descent_Algo
        alpha = self.fmodel.current_model_params["alpha"]
        beta = self.fmodel.current_model_params["beta"]
        gamma = self.fmodel.current_model_params["gamma"]
        u, s, v = self._get_svd_mini(gamma)
        self.log("model_deet alpha %s" % alpha)
        self.log("model_deet beta %s" % beta)
        self.log("model_deet sigma %s" % np.diag(s))

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

    def _get_lambda_derivatives(self):
        # override the function in Gradient_Descent_Algo
        # wrapper around calculating lambda derivatives
        alpha = self.fmodel.current_model_params["alpha"]
        beta = self.fmodel.current_model_params["beta"]
        gamma = self.fmodel.current_model_params["gamma"]

        u_hat, sigma_hat, v_hat = self._get_svd_mini(gamma)

        alpha, beta, row_features, col_features = self._get_nonzero_mini(alpha, beta)

        self.log("alpha %s" % alpha)
        self.log("beta %s" % beta)
        self.log("sigma_hat %s" % np.diag(sigma_hat))

        imp_derivs = Lamdba_Deriv_Problem_Wrapper(
            alpha,
            beta,
            u_hat,
            sigma_hat,
            v_hat,
        )

        dval_dlambda = []
        for i in range(self.fmodel.current_lambdas.size):
            self.log("SOLVING LAMBDA %d" % i)
            grad_dict_i = self._get_dmodel_dlambda(
                i,
                imp_derivs,
                alpha,
                beta,
                gamma,
                row_features,
                col_features,
                u_hat,
                sigma_hat,
                v_hat,
                self.fmodel.current_lambdas,
            )
            for k, v in grad_dict_i.iteritems():
                self.log("grad_dict %d: %s %s" % (i, k, v))
            dval_dlambda_i = self._get_val_gradient(
                grad_dict_i,
                alpha,
                beta,
                gamma,
                row_features,
                col_features
            )
            dval_dlambda.append(dval_dlambda_i)
        return np.array(dval_dlambda).flatten()

    def _get_vec_mask(self, idx):
        # returns a diagonal matrix multiplier that masks the specified indices
        vec_mask = np.zeros(self.settings.num_rows * self.settings.num_cols)
        vec_mask[idx] = 1
        return np.matrix(vec_mask).T

    # @print_time
    def _get_svd(self, gamma):
        # zeros out the singular values if close to zero
        # also transpose v
        u, s, v = np.linalg.svd(gamma)
        u_hat = u
        sigma_hat = np.diag((np.abs(s) > self.zero_thres) * s)
        v_hat = v.T
        return u_hat, sigma_hat, v_hat

    # @print_time
    def _get_svd_mini(self, gamma):
        # similar to _get_svd, but also
        # drops the zero singular values and the corresponding u and v columns
        u, s, v = np.linalg.svd(gamma)
        u_hat = u
        nonzero_mask = (np.abs(s) > self.zero_thres)
        u_mini = u[:, nonzero_mask]
        v_mini = v[nonzero_mask, :].T
        sigma_mini = np.diag(s[nonzero_mask])
        return u_mini, sigma_mini, v_mini

    def _get_nonzero_mini(self, alpha, beta):
        # return a smaller version of alpha and beta with zero elements removed
        # also returns a smaller version of the feature vectors
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
        # get gradient of the validation loss wrt lambda given the gradient of the
        # model parameters wrt lambda
        model_grad = grad_dict["dgamma_dlambda"]
        if alpha.size > 0:
            model_grad += row_features * grad_dict["dalpha_dlambda"] * self.onesT_row
        if beta.size > 0:
            model_grad += (col_features * grad_dict["dbeta_dlambda"] * self.onesT_col).T

        dval_dlambda = - 1.0/self.num_val * make_column_major_flat(
            self.data.observed_matrix
            - get_matrix_completion_fitted_values(
                row_features,
                col_features,
                alpha,
                beta,
                gamma,
            )
        )[self.data.validate_idx].T * make_column_major_flat(model_grad)[self.data.validate_idx]
        return dval_dlambda

    def _create_sigma_mask(self, sigma_hat):
        # mask with a zero along diagonal where sigma_hat is zero.
        # everywhere else is a one
        sigma_mask = np.ones(sigma_hat.shape)
        for i in range(sigma_hat.shape[0]):
            if sigma_hat[i,i] == 0:
                sigma_mask[i,i] = 0
        sigma_mask = make_column_major_flat(sigma_mask)
        return np.diag(sigma_mask.flatten())

    # @print_time
    def _get_d_square_loss(self, alpha, beta, gamma, row_features, col_features):
        # get first derivative of the square loss wrt X = gamma + stuff
        d_square_loss = - 1.0/self.num_train * np.multiply(
            self.train_vec,
            make_column_major_flat(
                self.data.observed_matrix
                - get_matrix_completion_fitted_values(
                    row_features,
                    col_features,
                    alpha,
                    beta,
                    gamma,
                )
            )
        )
        return d_square_loss

    # @print_time
    def _get_dd_square_loss(self, imp_derivs, row_features, col_features):
        # get double derivative of the square loss wrt X = gamma + stuff
        # imp_derivs should be a Lamdba_Deriv_Problem_Wrapper instance
        dd_square_loss = imp_derivs.dgamma_dlambda
        if imp_derivs.dalpha_dlambda is not None:
            dd_square_loss += row_features * imp_derivs.dalpha_dlambda * self.onesT_row
        if imp_derivs.dbeta_dlambda is not None:
            dd_square_loss += (col_features * imp_derivs.dbeta_dlambda * self.onesT_col).T
        dd_square_loss = 1.0/self.num_train * mul_elemwise(self.train_vec, vec(dd_square_loss))
        return dd_square_loss

    def _double_check_derivative_indepth(self, lambda_idx, model1, model2, model0, eps):
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

        print "dalpha_dlambda, %s" % (dalpha_dlambda)
        print "dBeta_dlambda, %s" % (dbeta_dlambda)
        print "dU_dlambda", dU_dlambda
        print "ds_dlambda, %s" % (dSigma_dlambda)
        # print "dgamma_dlambda, %s" % (dgamma_dlambda)

    def _check_optimality_conditions(self, model_params, lambdas, opt_thres=1e-2):
        # sanity check function to see that cvxpy is solving to a good enough accuracy
        # check that the gradient is close to zero
        # can use this to check that our implicit derivative assumptions hold
        # lambdas must be an exploded lambda matrix
        print "check_optimality_conditions!"

        alpha = model_params["alpha"]
        beta = model_params["beta"]
        gamma = model_params["gamma"]

        u_hat, sigma_hat, v_hat = self._get_svd_mini(gamma)

        d_square_loss = -1.0/self.num_train * np.multiply(
            self.train_vec,
            make_column_major_flat(
                self.data.observed_matrix
                - get_matrix_completion_fitted_values(
                    self.data.row_features,
                    self.data.col_features,
                    alpha,
                    beta,
                    gamma
                )
            )
        )

        left_grad_at_opt_gamma = (
            u_hat.T * make_column_major_reshape(d_square_loss, (self.data.num_rows, self.data.num_cols))
            + lambdas[0] * np.sign(sigma_hat) * v_hat.T
        )
        right_grad_at_opt_gamma = (
            make_column_major_reshape(d_square_loss, (self.data.num_rows, self.data.num_cols)) * v_hat
            + lambdas[0] * u_hat * np.sign(sigma_hat)
        )
        left_grad_norm = np.linalg.norm(left_grad_at_opt_gamma)
        right_grad_norm = np.linalg.norm(right_grad_at_opt_gamma)
        print "grad_at_opt wrt gamma (should be zero)", left_grad_norm, right_grad_norm
        assert(left_grad_norm < opt_thres)
        assert(right_grad_norm < opt_thres)

        print "alpha", alpha
        grad_at_opt_alpha = []
        for i in range(alpha.size):
            if np.abs(alpha[i]) > self.zero_thres:
                alpha_sign = np.sign(alpha[i])
                grad_at_opt_alpha.append((
                    d_square_loss.T * make_column_major_flat(
                        self.data.row_features[:, i] * self.onesT_row
                    )
                    + lambdas[1] * alpha_sign
                    + lambdas[2] * alpha[i]
                )[0,0])
        print "grad_at_opt wrt alpha (should be zero)", grad_at_opt_alpha
        assert(np.all(np.abs(grad_at_opt_alpha) < opt_thres))

        print "beta", beta
        grad_at_opt_beta = []
        for i in range(beta.size):
            if np.abs(beta[i]) > self.zero_thres:
                beta_sign = np.sign(beta[i])
                grad_at_opt_beta.append((
                    d_square_loss.T * make_column_major_flat(
                        (self.data.col_features[:, i] * self.onesT_col).T
                    )
                    + lambdas[3] * beta_sign
                    + lambdas[4] * beta[i]
                )[0,0])
        print "grad_at_opt wrt beta (should be zero)", grad_at_opt_beta
        assert(np.all(np.abs(grad_at_opt_beta) < opt_thres))

class Matrix_Completion_Hillclimb(Matrix_Completion_Hillclimb_Base):
    method_label = "Matrix_Completion_Hillclimb"

    def _create_lambda_configs(self):
        # have a bigger nuclear norm lambda parameter because otherwise calculating
        # the derivative is really slow
        self.lambda_mins = [5 * 1e-4] + [1e-6] * 4

    def _create_problem_wrapper(self):
        # self.problem_wrapper = MatrixCompletionProblemWrapper(self.data)
        self.problem_wrapper = MatrixCompletionProblemWrapperCustom(self.data)

    def _get_dmodel_dlambda(
            self,
            lambda_idx,
            imp_derivs,
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
        # this fcn accepts mini-fied model parameters - alpha, beta, and u/sigma/v
        # returns the gradient of the model parameters wrt lambda
        d_square_loss = self._get_d_square_loss(alpha, beta, gamma, row_features, col_features)
        d_square_loss_reshape = make_column_major_reshape(d_square_loss, (self.data.num_rows, self.data.num_cols))
        dd_square_loss = self._get_dd_square_loss(imp_derivs, row_features, col_features)
        dd_square_loss_reshape = reshape(
            dd_square_loss,
            self.data.num_rows,
            self.data.num_cols,
        )
        sigma_mask = self._create_sigma_mask(sigma_hat)
        obj = 0

        # Constraint from implicit differentiation of the optimality conditions
        # that were defined by taking the gradient of the training objective wrt gamma
        constraints_dgamma = []
        if sigma_hat.size > 0:
            # left multiply U^T and implicit derivative
            dgamma_left_imp_deriv_dlambda = (
                imp_derivs.dU_dlambda.T * d_square_loss_reshape
                + u_hat.T * dd_square_loss_reshape
                + lambdas[0] * np.sign(sigma_hat) * imp_derivs.dV_dlambda.T
            )

            # right multiply V and implicit derivative
            dgamma_right_imp_deriv_dlambda = (
                d_square_loss_reshape * imp_derivs.dV_dlambda
                + dd_square_loss_reshape * v_hat
                + lambdas[0] * imp_derivs.dU_dlambda * np.sign(sigma_hat)
            )
            if lambda_idx == 0:
                dgamma_left_imp_deriv_dlambda += np.sign(sigma_hat) * v_hat.T
                dgamma_right_imp_deriv_dlambda += u_hat * np.sign(sigma_hat)

            constraints_dgamma = [
                dgamma_left_imp_deriv_dlambda == 0,
                dgamma_right_imp_deriv_dlambda == 0
            ]
            obj += sum_squares(dgamma_left_imp_deriv_dlambda) + sum_squares(dgamma_right_imp_deriv_dlambda)

        # Constraint from implicit differentiation of the optimality conditions
        # that were defined by taking the gradient of the training objective wrt
        # alpha and beta, respectively

        constraints_dalpha = []
        for i in range(alpha.size):
            dalpha_imp_deriv_dlambda = (
                dd_square_loss.T * vec(row_features[:, i] * self.onesT_row)
                + lambdas[2] * imp_derivs.dalpha_dlambda[i]
            )
            if lambda_idx == 1:
                dalpha_imp_deriv_dlambda += np.sign(alpha[i])
            elif lambda_idx == 2:
                dalpha_imp_deriv_dlambda += alpha[i]
            constraints_dalpha.append(dalpha_imp_deriv_dlambda == 0)
            obj += sum_squares(dalpha_imp_deriv_dlambda)

        constraints_dbeta = []
        for i in range(beta.size):
            dbeta_imp_deriv_dlambda = (
                dd_square_loss.T * vec(
                    (col_features[:, i] * self.onesT_col).T
                )
                + lambdas[4] * imp_derivs.dbeta_dlambda[i]
            )
            if lambda_idx == 3:
                dbeta_imp_deriv_dlambda += np.sign(beta[i])
            elif lambda_idx == 4:
                dbeta_imp_deriv_dlambda += beta[i]
            constraints_dbeta.append(dbeta_imp_deriv_dlambda == 0)
            obj += sum_squares(dbeta_imp_deriv_dlambda)

        return imp_derivs.solve(constraints_dgamma + constraints_dalpha + constraints_dbeta, obj)

class Matrix_Completion_Hillclimb_Simple(Matrix_Completion_Hillclimb_Base):
    method_label = "Matrix_Completion_Hillclimb_Simple"

    def _create_lambda_configs(self):
        # have a bigger nuclear norm lambda parameter because otherwise calculating
        # the derivative is really slow
        self.lambda_mins = [5 * 1e-4, 1e-6]

    def _create_problem_wrapper(self):
        self.problem_wrapper = MatrixCompletionProblemWrapperSimple(self.data)

    def _check_optimality_conditions(self, model_params, lambdas):
        return
        # exploded_lambdas = [lambdas[0]] + [lambdas[1]] * 4
        # return super(Matrix_Completion_Hillclimb_Simple, self)._check_optimality_conditions(
        #     model_params,
        #     exploded_lambdas
        # )

    def _get_dmodel_dlambda(
            self,
            lambda_idx,
            imp_derivs,
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
        # TODO: make this same as above. this is wrong right now and will crash - solve with obejctive function, not constraints
        d_square_loss = self._get_d_square_loss(alpha, beta, gamma, row_features, col_features)
        d_square_loss_reshape = make_column_major_reshape(d_square_loss, (self.data.num_rows, self.data.num_cols))
        dd_square_loss = self._get_dd_square_loss(imp_derivs, row_features, col_features)
        dd_square_loss_reshape = reshape(
            dd_square_loss,
            self.data.num_rows,
            self.data.num_cols,
        )
        sigma_mask = self._create_sigma_mask(sigma_hat)

        # Constraint from implicit differentiation of the optimality conditions
        # that were defined by taking the gradient of the training objective wrt gamma
        constraints_dgamma = []
        if sigma_hat.size > 0:
            dgamma_imp_deriv_dlambda = (
                imp_derivs.dU_dlambda.T * d_square_loss_reshape * v_hat
                + u_hat.T * dd_square_loss_reshape * v_hat
                + u_hat.T * d_square_loss_reshape * imp_derivs.dV_dlambda
            )
            if lambda_idx == 0:
                dgamma_imp_deriv_dlambda += np.sign(sigma_hat)

            constraints_dgamma = [
                 sigma_mask * vec(dgamma_imp_deriv_dlambda) == np.zeros((self.data.num_rows * self.data.num_cols, 1))
            ]

        def _make_alpha_constraint(i):
            dalpha_imp_deriv_dlambda = (
                dd_square_loss.T * vec(row_features[:, i] * self.onesT_row)
                + lambdas[1] * imp_derivs.dalpha_dlambda[i]
            )
            if lambda_idx == 1:
                dalpha_imp_deriv_dlambda += np.sign(alpha[i]) + alpha[i]
            return dalpha_imp_deriv_dlambda == 0

        def _make_beta_constraint(i):
            dbeta_imp_deriv_dlambda = (
                dd_square_loss.T * vec(
                    (col_features[:, i] * self.onesT_col).T
                )
                + lambdas[1] * imp_derivs.dbeta_dlambda[i]
            )
            if lambda_idx == 1:
                dbeta_imp_deriv_dlambda += np.sign(beta[i]) + beta[i]
            return dbeta_imp_deriv_dlambda == 0

        # Constraint from implicit differentiation of the optimality conditions
        # that were defined by taking the gradient of the training objective wrt
        # alpha and beta, respectively
        constraints_dalpha = [_make_alpha_constraint(i) for i in range(alpha.size)]
        constraints_dbeta = [_make_beta_constraint(i) for i in range(beta.size)]

        return imp_derivs.solve(constraints_dgamma + constraints_dalpha + constraints_dbeta)

    def _double_check_derivative_indepth(self, i, model1, model2, model0, eps):
        # sanity check function for checking derivatives
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
