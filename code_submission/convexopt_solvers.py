from cvxpy import *
import cvxopt
from common import *
import scipy as sp
import cvxpy

SCS_MAX_ITERS = 10000
SCS_EPS = 1e-3
SCS_HIGH_ACC_EPS = 1e-6
ECOS_TOL = 1e-12
REALDATA_MAX_ITERS = 4000

class Lambda12ProblemWrapper:
    def __init__(self, X, y):
        n = X.shape[1]
        self.beta = Variable(n)
        self.lambda1 = Parameter(sign="positive")
        self.lambda2 = Parameter(sign="positive")
        objective = Minimize(
            0.5 * sum_squares(y - X * self.beta)
            + self.lambda1 * norm(self.beta, 1)
            + 0.5 * self.lambda2 * sum_squares(self.beta)
        )
        self.problem = Problem(objective, [])

    def solve(self, lambda1, lambda2):
        self.lambda1.value = lambda1
        self.lambda2.value = lambda2
        result = self.problem.solve(solver=SCS, verbose=VERBOSE)
        return self.beta.value

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

    def solve(self, lambdas):
        for idx in self.group_range:
            self.lambda1s[idx].value = lambdas[idx]

        self.lambda2.value = lambdas[-1]

        tol = 1e-6
        ecos_iters = 200
        try:
            self.problem.solve(solver=ECOS, verbose=VERBOSE, abstol=ECOS_TOL, reltol=ECOS_TOL, abstol_inacc=tol, reltol_inacc=tol, max_iters=ecos_iters)
        except SolverError:
            # switching to SCS if ECOS does not find a solution
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

    def solve(self, lambdas, high_accur=True):
        self.lambda1.value = lambdas[0]
        self.lambda2.value = lambdas[1]

        if high_accur:
            tol = 1e-6
            ecos_iters = 200
            try:
                self.problem.solve(solver=ECOS, verbose=VERBOSE, reltol=tol, abstol_inacc=tol, reltol_inacc=tol, max_iters=ecos_iters)
            except SolverError:
                print "switching to SCS!"
                self.problem.solve(solver=SCS, verbose=VERBOSE, eps=SCS_HIGH_ACC_EPS, max_iters=SCS_MAX_ITERS, use_indirect=False, normalize=False, warm_start=True)
        else:
            self.problem.solve(solver=SCS, verbose=VERBOSE)
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

class SmoothAndLinearProblemWrapper:
    # Use the penalized regression for APLM 3 - penalty for second order difference and an elastic net

    def __init__(self, X_linear, X_smooth, train_indices, y):
        # X_smooth must be an ordered matrix
        self.X_linear = X_linear
        self.y = y
        assert(np.array_equal(X_smooth, np.sort(X_smooth, axis=0)))

        feature_size = X_linear.shape[1]
        num_samples = X_smooth.size

        # Create second order difference matrix
        off_diag_D1 = [1] * (num_samples - 1)
        mid_diag_D1 = off_diag_D1 + [0]
        simple_d1 = np.matrix(np.diagflat(off_diag_D1, 1) - np.diagflat(mid_diag_D1))
        mid_diag = [1.0 / (X_smooth[i + 1, 0] - X_smooth[i, 0]) for i in range(0, num_samples - 1)] + [0]
        difference_matrix = simple_d1 * np.matrix(np.diagflat(mid_diag)) * simple_d1
        self.D = sp.sparse.coo_matrix(difference_matrix)
        D_sparse = cvxopt.spmatrix(self.D.data, self.D.row.tolist(), self.D.col.tolist())

        # Generate n x 1 boolean matrix to indicate which values in X_smooth belong to train set
        train_identity_matrix = np.matrix(np.eye(num_samples))[train_indices, :]
        self.train_eye = sp.sparse.coo_matrix(train_identity_matrix)
        train_identity_matrix_sparse = cvxopt.spmatrix(self.train_eye.data, self.train_eye.row.tolist(), self.train_eye.col.tolist())

        self.beta = Variable(feature_size)
        self.theta = Variable(num_samples)
        max_theta_idx = np.amax(np.where(train_indices)) + 1

        self.lambdas = [Parameter(sign="positive"), Parameter(sign="positive"), Parameter(sign="positive")]

        objective = (
            0.5 * sum_squares(y - X_linear * self.beta - train_identity_matrix_sparse * self.theta[0:max_theta_idx])
            + self.lambdas[0] * norm(self.beta, 1)
            + 0.5 * self.lambdas[1] * sum_squares(self.beta)
            + 0.5 * self.lambdas[2] * sum_squares(D_sparse * self.theta)
        )

        self.problem = Problem(Minimize(objective), [])

    def solve(self, lambdas):
        for i in range(0, len(lambdas)):
            self.lambdas[i].value = lambdas[i]

        result = self.problem.solve(solver=SCS, verbose=VERBOSE, max_iters=SCS_MAX_ITERS, use_indirect=False, normalize=True)

        return self.beta.value, self.theta.value

class SmoothAndLinearProblemWrapperSimple:
    # Use the penalized regression for APLM 2 - penalty for second order difference and a lasso penalty

    # X_smooth must be an ordered matrix
    def __init__(self, X_linear, X_smooth, train_indices, y):
        assert(np.array_equal(X_smooth, np.sort(X_smooth, axis=0)))

        feature_size = X_linear.shape[1]
        num_samples = X_smooth.size

        # Create second order difference matrix
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
        objective = (
            0.5 * sum_squares(y - X_linear * self.beta - train_matrix_sparse * self.theta[0:max_theta_idx])
            + self.lambdas[0] * norm(self.beta, 1)
            + 0.5 * self.lambdas[1] * sum_squares(D_sparse * self.theta)
        )
        self.problem = Problem(Minimize(objective), [])

    def solve(self, lambdas):
        for i in range(0, len(lambdas)):
            self.lambdas[i].value = lambdas[i]

        result = self.problem.solve(solver=SCS, verbose=VERBOSE, max_iters=SCS_MAX_ITERS, use_indirect=False, normalize=True)
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

    # @param high_accur: for gradient descent on the validation errors, accuracy is important
    def solve(self, lambdas, high_accur=True, warm_start=True):
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

        try:
            self.problem.solve(solver=ECOS, verbose=VERBOSE, abstol=ECOS_TOL, reltol=ECOS_TOL)
        except SolverError:
            # switching to SCS if ECOS has trouble finding a solution
            self.problem.solve(solver=SCS, verbose=VERBOSE, max_iters=max_iters, use_indirect=False, eps=eps, normalize=False, warm_start=warm_start)

        self.lambdas = lambdas
        self.thetas = thetas.value
        return thetas.value
