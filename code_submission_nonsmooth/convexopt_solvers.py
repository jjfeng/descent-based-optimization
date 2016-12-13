import time
from cvxpy import *
import cvxopt
from common import *
import scipy as sp
import cvxpy

SCS_MAX_ITERS = 10000
SCS_EPS = 1e-3 # default eps
SCS_HIGH_ACC_EPS = 1e-6
ECOS_TOL = 1e-12
REALDATA_MAX_ITERS = 4000

# Objective function: 0.5 * norm(y - Xb)^2 + lambda1 * lasso + 0.5 * lambda2 * ridge
class Lambda12ProblemWrapper:
    def __init__(self, X, y):
        n = X.shape[1]
        self.beta = Variable(n)
        self.lambda1 = Parameter(sign="positive")
        self.lambda2 = Parameter(sign="positive")
        objective = Minimize(0.5 * sum_squares(y - X * self.beta)
            + self.lambda1 * norm(self.beta, 1)
            + 0.5 * self.lambda2 * sum_squares(self.beta))
        self.problem = Problem(objective, [])

    def solve(self, lambda1, lambda2, quick_run=None):
        self.lambda1.value = lambda1
        self.lambda2.value = lambda2
        result = self.problem.solve(solver=SCS, verbose=VERBOSE)
        # print "self.problem.status", self.problem.status
        return self.beta.value

# Objective function: 0.5 * norm(y - Xb)^2 + lambda1 * lasso + 0.5 * lambda2 * ridge
class ElasticNetProblemWrapper:
    def __init__(self, X, y):
        n = X.shape[1]
        self.beta = Variable(n)
        self.lambda1 = Parameter(sign="positive")
        self.lambda2 = Parameter(sign="positive")
        objective = Minimize(0.5 * sum_squares(y - X * self.beta)
            + self.lambda1 * norm(self.beta, 1)
            + 0.5 * self.lambda2 * sum_squares(self.beta))
        self.problem = Problem(objective, [])

    def solve(self, lambdas, quick_run=None):
        self.lambda1.value = lambdas[0]
        self.lambda2.value = lambdas[1]
        result = self.problem.solve(solver=SCS, verbose=VERBOSE)
        return self.beta.value

# Objective function: 0.5 * norm(y - Xb)^2 + lambda * alpha * lasso + 0.5 * lambda * (1 - alpha) * ridge
# where alpha = exp(mu) / (1 + exp(mu))
class LambdaAlphaProblemWrapper:
    def __init__(self, X, y):
        n = X.shape[1]
        self.beta = Variable(n)
        self.lambda_param = Parameter(sign="positive")
        self.alpha_param = Parameter(sign="positive")
        self.one_minus_alpha_param = Parameter(sign="positive")
        objective = Minimize(0.5 * sum_squares(y - X * self.beta)
            + self.lambda_param * self.alpha_param * norm(self.beta, 1)
            + 0.5 * self.lambda_param * self.one_minus_alpha_param * sum_squares(self.beta))
        self.problem = Problem(objective, [])

    def solve(self, lambda_guess, mu_guess):
        alpha_guess = 1 - 1.0 / (1 + np.exp(mu_guess))
        self.lambda_param.value = lambda_guess
        self.alpha_param.value = alpha_guess
        self.one_minus_alpha_param.value = 1 - alpha_guess
        result = self.problem.solve(solver=SCS, verbose=VERBOSE)
        return self.beta.value

class EffectsInteractionProblemWrapper:
    def __init__(self, X, W, y):
        self.beta = Variable(X.shape[1])
        self.theta = Variable(W.shape[1])
        self.lambdas = [Parameter(sign="positive") for i in range(0, 4)]
        objective = Minimize(0.5 * sum_squares(y - X * self.beta - W * self.theta)
            + self.lambdas[0] * norm(self.beta, 1)
            + self.lambdas[1] * norm(self.theta, 1)
            + 0.5 * self.lambdas[2] * sum_squares(self.beta)
            + 0.5 * self.lambdas[3] * sum_squares(self.theta))
        self.problem = Problem(objective, [])

    def solve(self, lambdas):
        for i in range(0, 4):
            self.lambdas[i].value = lambdas[i]
        result = self.problem.solve(solver=SCS, verbose=VERBOSE)
        return self.beta.value, self.theta.value

    def solve_mu(self, mus):
        for i in range(0, 4):
            self.lambdas[i].value = np.exp(mus[i])
        result = self.problem.solve(solver=SCS, verbose=VERBOSE)
        return self.beta.value, self.theta.value

class EffectsInteractionProblemWrapperSimple:
    def __init__(self, X, W, y):
        self.beta = Variable(X.shape[1])
        self.theta = Variable(W.shape[1])
        self.lambdas = [Parameter(sign="positive") for i in range(0, 2)]
        objective = Minimize(0.5 * sum_squares(y - X * self.beta - W * self.theta)
            + self.lambdas[0] * (norm(self.beta, 1) + norm(self.theta, 1))
            + 0.5 * self.lambdas[1] * (sum_squares(self.beta) + sum_squares(self.theta)))
        self.problem = Problem(objective, [])

    def solve(self, lambdas):
        for i in range(0, 2):
            self.lambdas[i].value = lambdas[i]
        result = self.problem.solve(solver=SCS, verbose=VERBOSE)
        return self.beta.value, self.theta.value

class GroupedLassoProblemWrapper:
    def __init__(self, X, y, group_feature_sizes):
        self.group_range = range(0, len(group_feature_sizes))
        self.betas = [Variable(feature_size) for feature_size in group_feature_sizes]
        self.lambda1s = [Parameter(sign="positive") for i in self.group_range]
        self.lambda2 = Parameter(sign="positive")

        feature_start_idx = 0
        model_prediction = 0
        group_lasso_regularization = 0
        sparsity_regularization = 0
        for i in self.group_range:
            end_feature_idx = feature_start_idx + group_feature_sizes[i]
            model_prediction += X[:, feature_start_idx : end_feature_idx] * self.betas[i]
            feature_start_idx = end_feature_idx
            group_lasso_regularization += self.lambda1s[i] * norm(self.betas[i], 2)
            sparsity_regularization += norm(self.betas[i], 1)

        objective = Minimize(0.5 / y.size * sum_squares(y - model_prediction)
            + group_lasso_regularization
            + self.lambda2 * sparsity_regularization)
        self.problem = Problem(objective, [])

    def solve(self, lambdas, quick_run=False):
        for idx in self.group_range:
            self.lambda1s[idx].value = lambdas[idx]

        self.lambda2.value = lambdas[-1]

        ecos_iters = 200
        try:
            self.problem.solve(solver=ECOS, verbose=VERBOSE, abstol=ECOS_TOL, reltol=ECOS_TOL, abstol_inacc=ECOS_TOL, reltol_inacc=ECOS_TOL, max_iters=ecos_iters)
        except SolverError:
            self.problem.solve(solver=SCS, verbose=VERBOSE, eps=SCS_HIGH_ACC_EPS/100, max_iters=SCS_MAX_ITERS * 4, use_indirect=False, normalize=False, warm_start=True)

        return [b.value for b in self.betas]


class GroupedLassoClassifyProblemWrapper:
    def __init__(self, X_groups, y):
        group_feature_sizes = [g.shape[1] for g in X_groups]
        self.group_range = range(0, len(group_feature_sizes))
        self.betas = [Variable(feature_size) for feature_size in group_feature_sizes]
        self.lambda1s = [Parameter(sign="positive") for i in self.group_range]
        self.lambda2 = Parameter(sign="positive")

        feature_start_idx = 0
        model_prediction = 0
        group_lasso_regularization = 0
        sparsity_regularization = 0
        for i in self.group_range:
            model_prediction += X_groups[i] * self.betas[i]
            group_lasso_regularization += self.lambda1s[i] * norm(self.betas[i], 2)
            sparsity_regularization += norm(self.betas[i], 1)

        log_sum = 0
        for i in range(0, X_groups[0].shape[0]):
            one_plus_expyXb = vstack(0, model_prediction[i])
            log_sum += log_sum_exp(one_plus_expyXb)

        objective = Minimize(
            log_sum
            - model_prediction.T * y
            + group_lasso_regularization
            + self.lambda2 * sparsity_regularization)
        self.problem = Problem(objective, [])

    def solve(self, lambdas):
        for idx in self.group_range:
            self.lambda1s[idx].value = lambdas[idx]

        self.lambda2.value = lambdas[-1]

        result = self.problem.solve(solver=SCS, verbose=VERBOSE, max_iters=REALDATA_MAX_ITERS, use_indirect=False, normalize=True)
        print "self.problem.status", self.problem.status
        return [b.value for b in self.betas]


class GroupedLassoClassifyProblemWrapperFullCV:
    def __init__(self, X, y, group_feature_sizes):
        self.group_range = range(0, len(group_feature_sizes))
        total_features = np.sum(group_feature_sizes)
        self.beta = Variable(total_features)
        self.lambda1s = [Parameter(sign="positive") for i in self.group_range]
        self.lambda2 = Parameter(sign="positive")

        start_feature_idx = 0
        group_lasso_regularization = 0
        for i, group_feature_size in enumerate(group_feature_sizes):
            end_feature_idx = start_feature_idx + group_feature_size
            group_lasso_regularization += self.lambda1s[i] * norm(self.beta[start_feature_idx:end_feature_idx], 2)
            start_feature_idx = end_feature_idx

        model_prediction = X * self.beta
        log_sum = 0
        num_samples = X.shape[0]
        for i in range(0, num_samples):
            one_plus_expyXb = vstack(0, model_prediction[i])
            log_sum += log_sum_exp(one_plus_expyXb)

        objective = Minimize(
            log_sum
            - (X * self.beta).T * y
            + group_lasso_regularization
            + self.lambda2 * norm(self.beta, 1))
        self.problem = Problem(objective, [])

    def solve(self, lambdas):
        for idx in self.group_range:
            self.lambda1s[idx].value = lambdas[idx]

        self.lambda2.value = lambdas[-1]

        result = self.problem.solve(solver=SCS, verbose=VERBOSE, max_iters=REALDATA_MAX_ITERS, use_indirect=False, normalize=True)
        print "self.problem.status", self.problem.status
        return self.beta.value


class GroupedLassoProblemWrapperSimple:
    def __init__(self, X, y, group_feature_sizes):
        self.group_range = range(0, len(group_feature_sizes))
        self.betas = [Variable(feature_size) for feature_size in group_feature_sizes]
        self.lambda1 = Parameter(sign="positive")
        self.lambda2 = Parameter(sign="positive")

        feature_start_idx = 0
        model_prediction = 0
        group_lasso_regularization = 0
        sparsity_regularization = 0
        for i in self.group_range:
            end_feature_idx = feature_start_idx + group_feature_sizes[i]
            model_prediction += X[:, feature_start_idx : end_feature_idx] * self.betas[i]
            feature_start_idx = end_feature_idx
            group_lasso_regularization += norm(self.betas[i], 2)
            sparsity_regularization += norm(self.betas[i], 1)

        objective = Minimize(0.5 / y.size * sum_squares(y - model_prediction)
            + self.lambda1 * group_lasso_regularization
            + self.lambda2 * sparsity_regularization)
        self.problem = Problem(objective, [])

    def solve(self, lambdas, quick_run=False):
        self.lambda1.value = lambdas[0]
        self.lambda2.value = lambdas[1]

        if not quick_run:
            ecos_iters = 400
            tol=ECOS_TOL * 100
            try:
                self.problem.solve(solver=ECOS, verbose=VERBOSE, reltol=tol, abstol_inacc=tol, reltol_inacc=tol, max_iters=ecos_iters)
            except SolverError:
                print "switching to SCS!"
                self.problem.solve(solver=SCS, verbose=VERBOSE, eps=SCS_HIGH_ACC_EPS, max_iters=SCS_MAX_ITERS * 4, use_indirect=False, normalize=False, warm_start=True)
        else:
            try:
                self.problem.solve(solver=ECOS, verbose=VERBOSE)
            except SolverError:
                print "switching to SCS!"
                self.problem.solve(solver=SCS, verbose=VERBOSE, use_indirect=False, normalize=False, warm_start=True)
        return [b.value for b in self.betas]

class GroupedLassoClassifyProblemWrapperSimple:
    def __init__(self, X_groups, y):
        self.group_range = range(0, len(X_groups))
        self.betas = [Variable(Xg.shape[1]) for Xg in X_groups]
        self.lambda1 = Parameter(sign="positive")
        self.lambda2 = Parameter(sign="positive")

        feature_start_idx = 0
        model_prediction = 0
        group_lasso_regularization = 0
        sparsity_regularization = 0
        for i, Xg in enumerate(X_groups):
            model_prediction += Xg * self.betas[i]
            group_lasso_regularization += norm(self.betas[i], 2)
            sparsity_regularization += norm(self.betas[i], 1)

        log_sum = 0
        for i in range(0, X_groups[0].shape[0]):
            one_plus_expyXb = vstack(0, model_prediction[i])
            log_sum += log_sum_exp(one_plus_expyXb)

        objective = Minimize(
            log_sum
            - model_prediction.T * y
            + self.lambda1 * group_lasso_regularization
            + self.lambda2 * sparsity_regularization)
        self.problem = Problem(objective, [])

    def solve(self, lambdas):
        self.lambda1.value = lambdas[0]
        self.lambda2.value = lambdas[1]

        result = self.problem.solve(solver=SCS, verbose=VERBOSE)
        return [b.value for b in self.betas]


class GroupedLassoClassifyProblemWrapperSimpleFullCV:
    def __init__(self, X, y, feature_group_sizes):
        total_features = np.sum(feature_group_sizes)
        self.beta = Variable(total_features)
        self.lambda1 = Parameter(sign="positive")
        self.lambda2 = Parameter(sign="positive")

        start_feature_idx = 0
        group_lasso_regularization = 0
        for feature_group_size in feature_group_sizes:
            end_feature_idx = start_feature_idx + feature_group_size
            group_lasso_regularization += norm(self.beta[start_feature_idx:end_feature_idx], 2)
            start_feature_idx = end_feature_idx

        model_prediction = X * self.beta
        log_sum = 0
        num_samples = X.shape[0]
        for i in range(0, num_samples):
            one_plus_expyXb = vstack(0, model_prediction[i])
            log_sum += log_sum_exp(one_plus_expyXb)

        objective = Minimize(
            log_sum
            - model_prediction.T * y
            + self.lambda1 * group_lasso_regularization
            + self.lambda2 * norm(self.beta, 1))
        self.problem = Problem(objective, [])

    def solve(self, lambdas):
        self.lambda1.value = lambdas[0]
        self.lambda2.value = lambdas[1]

        result = self.problem.solve(solver=SCS, verbose=VERBOSE)
        return self.beta.value


class LassoClassifyProblemWrapper:
    def __init__(self, X, y, _):
        self.beta = Variable(X.shape[1])
        self.lambda1 = Parameter(sign="positive")

        model_prediction = X * self.beta

        log_sum = 0
        num_samples = X.shape[0]
        for i in range(0, num_samples):
            one_plus_expyXb = vstack(0, model_prediction[i])
            log_sum += log_sum_exp(one_plus_expyXb)

        objective = Minimize(
            log_sum
            - model_prediction.T * y
            + self.lambda1 * norm(self.beta, 1))
        self.problem = Problem(objective, [])

    def solve(self, lambda_guesses):
        self.lambda1.value = lambda_guesses[0]

        result = self.problem.solve(solver=SCS, verbose=VERBOSE)
        return self.beta.value


# solving min 0.5 * (y - X_linear * beta - theta) + lambda1 * norm1(beta) + 0.5 * lambda2 * norm2(beta)^2 + 0.5 * lambda3 * norm2(D*theta)^2
class SmoothAndLinearProblemWrapper:
    # X_smooth must be an ordered matrix
    # train_identity_matrix: n x 1 boolean matrix to indicate which values in X_smooth belong to train set
    def __init__(self, X_linear, X_smooth, train_indices, y, use_l1=False):
        self.X_linear = X_linear
        self.y = y
        assert(np.array_equal(X_smooth, np.sort(X_smooth, axis=0)))

        feature_size = X_linear.shape[1]
        num_samples = X_smooth.size

        # we want a 1st order trend filtering, so we want D^2, not D^1
        off_diag_D1 = [1] * (num_samples - 1)
        mid_diag_D1 = off_diag_D1 + [0]
        simple_d1 = np.matrix(np.diagflat(off_diag_D1, 1) - np.diagflat(mid_diag_D1))
        mid_diag = [1.0 / (X_smooth[i + 1, 0] - X_smooth[i, 0]) for i in range(0, num_samples - 1)] + [0]
        difference_matrix = simple_d1 * np.matrix(np.diagflat(mid_diag)) * simple_d1
        self.D = sp.sparse.coo_matrix(difference_matrix)
        D_sparse = cvxopt.spmatrix(self.D.data, self.D.row.tolist(), self.D.col.tolist())

        train_identity_matrix = np.matrix(np.eye(num_samples))[train_indices, :]
        self.train_eye = sp.sparse.coo_matrix(train_identity_matrix)
        train_identity_matrix_sparse = cvxopt.spmatrix(self.train_eye.data, self.train_eye.row.tolist(), self.train_eye.col.tolist())

        self.beta = Variable(feature_size)
        self.theta = Variable(num_samples)
        max_theta_idx = np.amax(np.where(train_indices)) + 1

        self.lambdas = [Parameter(sign="positive"), Parameter(sign="positive"), Parameter(sign="positive")]

        objective = (0.5 * sum_squares(y - X_linear * self.beta - train_identity_matrix_sparse * self.theta[0:max_theta_idx])
            + self.lambdas[0] * norm(self.beta, 1)
            + 0.5 * self.lambdas[1] * sum_squares(self.beta))
        if use_l1:
            objective += self.lambdas[2] * norm(D_sparse * self.theta, 1)
        else:
            objective += 0.5 * self.lambdas[2] * sum_squares(D_sparse * self.theta)

        self.problem = Problem(Minimize(objective), [])

    def solve(self, lambdas, use_robust=True):
        for i in range(0, len(lambdas)):
            self.lambdas[i].value = lambdas[i]

        result = self.problem.solve(solver=SCS, verbose=VERBOSE, max_iters=SCS_MAX_ITERS, use_indirect=False, normalize=True)

        print "self.problem.status", self.problem.status
        return self.beta.value, self.theta.value


# solving min 0.5 * (y - X_linear * beta - theta) + lambda1 * norm1(beta) + 0.5 * lambda3 * norm2(D*theta)^2
class SmoothAndLinearProblemWrapperSimple:
    # X_smooth must be an ordered matrix
    # train_identity_matrix: n x 1 boolean matrix to indicate which values in X_smooth belong to train set
    def __init__(self, X_linear, X_smooth, train_indices, y, use_l1=False):
        assert(np.array_equal(X_smooth, np.sort(X_smooth, axis=0)))

        feature_size = X_linear.shape[1]
        num_samples = X_smooth.size

        # we want a 1st order trend filtering, so we want D^2, not D^1
        off_diag_D1 = [1] * (num_samples - 1)
        mid_diag_D1 = off_diag_D1 + [0]
        simple_d1 = np.matrix(np.diagflat(off_diag_D1, 1) - np.diagflat(mid_diag_D1))
        mid_diag = [1.0 / (X_smooth[i + 1, 0] - X_smooth[i, 0]) for i in range(0, num_samples - 1)] + [0]
        self.D = sp.sparse.coo_matrix(simple_d1 * np.matrix(np.diagflat(mid_diag)) * simple_d1)
        D_sparse = cvxopt.spmatrix(self.D.data, self.D.row.tolist(), self.D.col.tolist())

        train_matrix = sp.sparse.coo_matrix(np.matrix(np.eye(num_samples))[train_indices, :])
        train_matrix_sparse = cvxopt.spmatrix(train_matrix.data, train_matrix.row.tolist(), train_matrix.col.tolist())
        max_theta_idx = np.amax(np.where(train_indices)) + 1

        self.beta = Variable(feature_size)
        self.theta = Variable(num_samples)
        self.lambdas = [Parameter(sign="positive"), Parameter(sign="positive")]
        objective = 0.5 * sum_squares(y - X_linear * self.beta - train_matrix_sparse * self.theta[0:max_theta_idx]) + self.lambdas[0] * norm(self.beta, 1)
        if use_l1:
            objective += self.lambdas[1] * norm(D_sparse * self.theta, 1)
        else:
            objective += 0.5 * self.lambdas[1] * sum_squares(D_sparse * self.theta)

        self.problem = Problem(Minimize(objective), [])

    def solve(self, lambdas):
        for i in range(0, len(lambdas)):
            self.lambdas[i].value = lambdas[i]

        result = self.problem.solve(solver=SCS, verbose=VERBOSE, max_iters=SCS_MAX_ITERS, use_indirect=False, normalize=True)
        print "self.problem.status", self.problem.status
        return self.beta.value, self.theta.value

# min 0.5 * |y - train_identifiers * theta|^2 + 0.5 * sum (lambda_j * |D_j * theta_j|^2) + e * sum (|theta_j|^2)
# We have three features, want a smooth fit
class GenAddModelProblemWrapper:
    def __init__(self, X, train_indices, y, tiny_e=0):
        self.tiny_e = tiny_e
        self.y = y

        num_samples, num_features = X.shape
        self.num_samples = num_samples
        self.num_features = num_features

        # Create smooth penalty matrix for each feature
        self.diff_matrices = []
        for i in range(num_features):
            x_features = X[:,i]
            d1_matrix = np.zeros((num_samples, num_samples))
            # 1st, figure out ordering of samples for the feature
            sample_ordering = np.argsort(x_features)
            ordered_x = x_features[sample_ordering]
            d1_matrix[range(num_samples - 1), sample_ordering[:-1]] = -1
            d1_matrix[range(num_samples - 1), sample_ordering[1:]] = 1
            inv_dists = 1.0 / (ordered_x[np.arange(1, num_samples)] - ordered_x[np.arange(num_samples - 1)])
            inv_dists = np.append(inv_dists, 0)

            # Check that the inverted distances are all greater than zero
            assert(np.min(inv_dists) >= 0)
            D = d1_matrix * np.matrix(np.diagflat(inv_dists)) * d1_matrix
            self.diff_matrices.append(D)

        self.train_indices = train_indices
        self.train_identifier = np.matrix(np.zeros((len(train_indices), num_samples)))
        self.num_train = len(train_indices)
        self.train_identifier[np.arange(self.num_train), train_indices] = 1

    # @param high_accur: for gradient descent on the validation errors, getting the optimal solution is super important.
    # We need it in order to have an accurate gradient for validation loss wrt lambdas
    # as dimension of the solution vector increases, the number of iterations of SCS is necessary!
    def solve(self, lambdas, high_accur=True, warm_start=True):
        #TODO: Why is the problem crated a new time per iteration?!
        thetas = Variable(self.num_samples, self.num_features)
        objective = 0.5/self.num_train * sum_squares(self.y - sum_entries(thetas[self.train_indices,:], axis=1))
        for i in range(len(lambdas)):
            D = sp.sparse.coo_matrix(self.diff_matrices[i])
            D_sparse = cvxopt.spmatrix(D.data, D.row.tolist(), D.col.tolist())
            objective += 0.5/self.num_samples * lambdas[i] * sum_squares(D_sparse * thetas[:,i])
        # objective += 0.5 * self.tiny_e/(self.num_features * self.num_samples) * sum_squares(thetas)
        self.problem = Problem(Minimize(objective))
        if high_accur:
            eps = SCS_HIGH_ACC_EPS
            max_iters = SCS_MAX_ITERS * 4 * self.num_features # 5 * num_features
        else:
            eps = SCS_EPS
            max_iters = SCS_MAX_ITERS * 2

        # Don't use ECOS/ECOS_BB - for some reason, it's not finding good minimizers of the fcn. Even though the gradient of the training loss
        # does reach zero, it doesn't match the calculated gradient for some reason. My guess is that ECOS is getting stuck somewhere.
        # Ignoring that, it seems to just change on reg parameter and ignore the other ones.
        #### HUHHHH NOW ITS WORKING?! WTF.

        # Using indirect does not work - bad derivatives! Not normalizing is also better - bigger changes.
        try:
            self.problem.solve(solver=ECOS, verbose=VERBOSE, abstol=ECOS_TOL, reltol=ECOS_TOL)
        except SolverError:
            print "switching to SCS!"
            self.problem.solve(solver=SCS, verbose=VERBOSE, max_iters=max_iters, use_indirect=False, eps=eps, normalize=False, warm_start=warm_start)

        print "cvxpy, self.problem.status", self.problem.status, "value", self.problem.value
        self.lambdas = lambdas
        self.thetas = thetas.value
        return thetas.value

    def get_cost_components(self):
        total_train_cost = self.problem.value
        num_train = self.y.size

        train_loss = 0.5/num_train * get_norm2(
            self.y - np.sum(self.thetas[self.train_indices,:], axis=1),
            power=2
        )
        penalties = []
        for i in range(self.num_features):
            theta = np.matrix(self.thetas[:,i])
            lam = self.lambdas[i]
            D = self.diff_matrices[i]
            penalties.append(
                .5/self.num_samples * lam * get_norm2(D * theta, power=2)
            )
        tiny_e_cost = 0.5 * self.tiny_e/(self.num_features * self.num_samples) * get_norm2(self.thetas, power=2)
        print "cost_components:"
        print "tot", total_train_cost
        print "t", train_loss
        print "p", penalties
        print "e", tiny_e_cost
        print "diff", np.abs(total_train_cost - train_loss - sum(penalties) - tiny_e_cost)
        assert(np.abs(total_train_cost - train_loss - sum(penalties) - tiny_e_cost)/total_train_cost < 0.05)
        return total_train_cost, train_loss, penalties, tiny_e_cost

class SparseAdditiveModelProblemWrapperSimple:
    # A two lambda version
    def __init__(self, X, train_indices, y, tiny_e=0):
        self.tiny_e = tiny_e
        self.y = y

        num_samples, num_features = X.shape
        self.num_samples = num_samples
        self.num_features = num_features

        # Create smooth penalty matrix for each feature
        self.diff_matrices = []
        for i in range(num_features):
            D = _make_discrete_diff_matrix_ord2(X[:,i])
            self.diff_matrices.append(D)

        self.train_indices = train_indices
        num_train = train_indices.size
        
        self.lambdas = [Parameter(sign="positive"), Parameter(sign="positive")]
        self.thetas = Variable(self.num_samples, self.num_features)
        objective = 0.5/num_train * sum_squares(self.y - sum_entries(self.thetas[self.train_indices,:], axis=1))
        # group-level sparsity penalty
        objective += sum([1.0/num_train * self.lambdas[0] * pnorm(self.thetas[:,i], 2) for i in range(self.num_features)])
        for i in range(len(self.diff_matrices)):
            # sparse matrices working on the cluster machines
            # D = sp.sparse.coo_matrix(self.diff_matrices[i])
            # D_sparse = cvxopt.spmatrix(D.data, D.row.tolist(), D.col.tolist())

            # individual sparsity penalty - same lambda
            objective += 1.0/num_train * self.lambdas[1] * pnorm(self.diff_matrices[i] * self.thetas[:,i], 1)
        objective += 0.5 * self.tiny_e * sum_squares(self.thetas)
        self.problem = Problem(Minimize(objective))

    def solve(self, lambdas, warm_start=True, quick_run=False):
        for i,l in enumerate(lambdas):
            self.lambdas[i].value = lambdas[i]

        self.problem.solve(solver=SCS, verbose=VERBOSE, warm_start=warm_start)
        return self.thetas.value

class SparseAdditiveModelProblemWrapper:
    def __init__(self, X, train_indices, y, tiny_e=0):
        self.tiny_e = tiny_e
        self.y = y

        num_samples, num_features = X.shape
        self.num_samples = num_samples
        self.num_features = num_features

        # Create smooth penalty matrix for each feature
        self.diff_matrices = []
        for i in range(num_features):
            D = _make_discrete_diff_matrix_ord2(X[:,i])
            self.diff_matrices.append(D)

        self.train_indices = train_indices
        self.lambdas = [Parameter(sign="positive")]
        for i in range(self.num_features):
            self.lambdas.append(Parameter(sign="positive"))

        self.thetas = Variable(self.num_samples, self.num_features)
        num_train = train_indices.size
        objective = 0.5/num_train * sum_squares(self.y - sum_entries(self.thetas[self.train_indices,:], axis=1))
        objective += sum([1.0/num_train * self.lambdas[0] * pnorm(self.thetas[:,i], 2) for i in range(self.num_features)])
        for i in range(len(self.diff_matrices)):
            # D = sp.sparse.coo_matrix(self.diff_matrices[i])
            # D_sparse = cvxopt.spmatrix(D.data, D.row.tolist(), D.col.tolist())
            objective += 1.0/num_train * self.lambdas[i + 1] * pnorm(self.diff_matrices[i] * self.thetas[:,i], 1)
        objective += 0.5/num_train * self.tiny_e * sum_squares(self.thetas)
        self.problem = Problem(Minimize(objective))

    # @param high_accur: for gradient descent on the validation errors, getting the optimal solution is super important.
    # We need it in order to have an accurate gradient for validation loss wrt lambdas
    # as dimension of the solution vector increases, the number of iterations of SCS is necessary!
    def solve(self, lambdas, high_accur=True, warm_start=True, quick_run=False):
        # start_time = time.time()
        # print "cvxpy solve"
        for i in range(lambdas.size):
            self.lambdas[i].value = lambdas[i]

        # ECOS is not providing good enough precision for some reason
        if high_accur and not quick_run:
            eps = SCS_HIGH_ACC_EPS * 1e-3
            max_iters = SCS_MAX_ITERS * 10
        else:
            eps = SCS_EPS
            max_iters = SCS_MAX_ITERS

        # Don't use ECOS/ECOS_BB - for some reason, it's not finding good minimizers of the fcn. Even though the gradient of the training loss
        # self.problem.solve(solver=ECOS, verbose=VERBOSE, abstol=ECOS_TOL, reltol=ECOS_TOL, max_iters=200)
        if quick_run:
            self.problem.solve(solver=SCS, verbose=VERBOSE, max_iters=max_iters, eps=eps, warm_start=warm_start)
        else:
            self.problem.solve(solver=SCS, verbose=VERBOSE, max_iters=max_iters, use_indirect=False, eps=eps, normalize=False, warm_start=warm_start)

        # print "cvxpy, self.problem.status", self.problem.status, "value", self.problem.value
        #print "cvxpy runtime", time.time() - start_time

        if self.problem.value > 0 and self.problem.status in [OPTIMAL,  OPTIMAL_INACCURATE]:
            return self.thetas.value
        else:
            if self.problem.value < 0:
                print "Warning: Negative problem solution from cvxpy"
            return None

def _make_discrete_diff_matrix_ord2(x_features):
    num_samples = len(x_features)
    d1_matrix = np.matrix(np.zeros((num_samples, num_samples)))
    # 1st, figure out ordering of samples for the feature
    sample_ordering = np.argsort(x_features)
    ordered_x = x_features[sample_ordering]
    d1_matrix[range(num_samples - 1), sample_ordering[:-1]] = -1
    d1_matrix[range(num_samples - 1), sample_ordering[1:]] = 1
    inv_dists = 1.0 / (ordered_x[np.arange(1, num_samples)] - ordered_x[np.arange(num_samples - 1)])
    inv_dists = np.append(inv_dists, 0)

    # Check that the inverted distances are all greater than zero
    assert(np.min(inv_dists) >= 0)
    D = d1_matrix * np.matrix(np.diagflat(inv_dists)) * d1_matrix
    return D
