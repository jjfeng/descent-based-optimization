import numpy as np

from common import *

class ObservedData:
    def __init__(self, X_train, y_train, X_validate, y_validate, X_test, y_test):
        self.num_features = X_train.shape[1]

        self.X_train = X_train
        self.y_train = y_train
        self.X_validate = X_validate
        self.y_validate = y_validate
        self.X_test = X_test
        self.y_test = y_test

        self.num_train = y_train.size
        self.num_validate = y_validate.size
        self.num_test = y_test.size
        self.num_samples = self.num_train + self.num_validate + self.num_test

        self.X_full = np.vstack((X_train, X_validate, X_test))

        self.train_idx = np.arange(0, self.num_train)
        self.validate_idx = np.arange(self.num_train, self.num_train + self.num_validate)
        self.test_idx = np.arange(self.num_train + self.num_validate, self.num_train + self.num_validate + self.num_test)

class MatrixObservedData:
    def __init__(self, row_features, col_features, train_idx, validate_idx, test_idx, observed_matrix, alpha, beta, gamma, real_matrix):
        self.num_rows = row_features.shape[0]
        self.num_row_features = row_features.shape[1]
        self.num_cols = col_features.shape[0]
        self.num_col_features = col_features.shape[1]

        self.row_features = row_features
        self.col_features = col_features
        self.train_idx = train_idx
        self.validate_idx = validate_idx
        self.test_idx = test_idx
        self.observed_matrix = observed_matrix

        self.real_matrix = real_matrix
        self.real_alpha = alpha
        self.real_beta = beta
        self.real_gamma = gamma

class MatrixGroupsObservedData:
    def __init__(self, row_features, col_features, train_idx, validate_idx, test_idx, observed_matrix, alphas, betas, gamma, real_matrix):
        self.num_rows = real_matrix.shape[0]
        self.num_cols = real_matrix.shape[1]

        self.num_alphas = len(alphas)
        self.num_betas = len(betas)

        self.row_features = row_features
        self.col_features = col_features
        self.train_idx = train_idx
        self.validate_idx = validate_idx
        self.test_idx = test_idx
        self.observed_matrix = observed_matrix

        self.real_matrix = real_matrix
        self.real_alphas = alphas
        self.real_betas = betas
        self.real_gamma = gamma

class DataGenerator:
    def __init__(self, settings):
        self.settings = settings
        self.train_size = settings.train_size
        self.validate_size = settings.validate_size
        self.test_size = settings.test_size
        self.total_samples = settings.train_size + settings.validate_size + settings.test_size
        self.snr = settings.snr
        self.feat_range = settings.feat_range

    def make_additive_smooth_data(self, smooth_fcn_list):
        self.num_features = len(smooth_fcn_list)
        all_Xs = map(lambda x: self._make_shuffled_uniform_X(), range(self.num_features))
        X_smooth = np.column_stack(all_Xs)

        y_smooth = 0
        for idx, fcn in enumerate(smooth_fcn_list):
            y_smooth += fcn(X_smooth[:, idx]).reshape(self.total_samples, 1)

        return self._make_data(y_smooth, X_smooth)

    def make_correlated(self, num_features, num_nonzero_features):
        self.num_features = num_features
        # Multiplying by the cholesky decomposition of the covariance matrix should suffice: http://www.sitmo.com/article/generating-correlated-random-numbers/
        correlation_matrix = np.matrix([[np.power(0.5, abs(i - j)) for i in range(0, num_features)] for j in range(0, num_features)])
        X = np.matrix(np.random.randn(self.total_samples, num_features)) * np.matrix(np.linalg.cholesky(correlation_matrix)).T

        # beta real is a shuffled array of zeros and iid std normal values
        beta_real = np.matrix(
            np.concatenate((
                np.ones((num_nonzero_features, 1)),
                np.zeros((num_features - num_nonzero_features, 1))
            ))
        )
        np.random.shuffle(beta_real)

        true_y = X * beta_real
        data = self._make_data(true_y, X)
        data.beta_real = beta_real
        return data

    def sparse_groups(self, base_nonzero_coeff=[1, 2, 3, 4, 5]):
        group_feature_sizes = self.settings.get_true_group_sizes()
        nonzero_features = len(base_nonzero_coeff)

        X = np.matrix(np.random.randn(self.total_samples, np.sum(group_feature_sizes)))
        betas = [
            np.matrix(np.concatenate((base_nonzero_coeff, np.zeros(num_features - nonzero_features)))).T
            for num_features in group_feature_sizes
        ]
        beta = np.matrix(np.concatenate(betas))

        true_y = X * beta
        data = self._make_data(true_y, X)
        data.beta_real = beta
        return data

    def matrix_completion(self, alpha_val=1, beta_val=1, sv_val=1):
        def _make_correlation_matrix(cor_factor, num_feat):
            correlation_matrix = np.matrix([[np.power(cor_factor, abs(i - j)) for i in range(0, num_feat)] for j in range(0, num_feat)])
            return np.matrix(np.linalg.cholesky(correlation_matrix)).T

        matrix_shape = (self.settings.num_rows, self.settings.num_cols)

        alpha = np.matrix(np.concatenate((
            alpha_val * np.ones(self.settings.num_nonzero_row_features),
            np.zeros(self.settings.num_row_features - self.settings.num_nonzero_row_features)
        ))).T
        beta = np.matrix(np.concatenate((
            beta_val * np.ones(self.settings.num_nonzero_col_features),
            np.zeros(self.settings.num_col_features - self.settings.num_nonzero_col_features)
        ))).T

        row_corr_matrix = _make_correlation_matrix(0.5, self.settings.num_row_features)
        col_corr_matrix = _make_correlation_matrix(0.1, self.settings.num_col_features)

        row_features = np.matrix(np.random.randn(self.settings.num_rows, self.settings.num_row_features)) * row_corr_matrix
        col_features = np.matrix(np.random.randn(self.settings.num_cols, self.settings.num_col_features)) * col_corr_matrix

        gamma = 0
        for i in range(self.settings.num_nonzero_s):
            u = np.random.multivariate_normal(np.zeros(self.settings.num_rows), np.eye(self.settings.num_rows))
            v = np.random.multivariate_normal(np.zeros(self.settings.num_cols), np.eye(self.settings.num_cols))
            gamma += sv_val * np.matrix(u).T * np.matrix(v)

        true_matrix = get_matrix_completion_fitted_values(
            row_features,
            col_features,
            alpha,
            beta,
            gamma,
        )

        epsilon = np.random.randn(matrix_shape[0], matrix_shape[1])
        SNR_factor = self._make_snr_factor(np.linalg.norm(true_matrix, ord="fro"), np.linalg.norm(epsilon))
        observed_matrix = true_matrix + 1.0 / SNR_factor * epsilon

        # index column-major style
        shuffled_idx = np.random.permutation(matrix_shape[0] * matrix_shape[1])
        train_indices = shuffled_idx[0:self.settings.train_size]
        validate_indices = shuffled_idx[self.settings.train_size:self.settings.train_size + self.settings.validate_size]
        test_indices = shuffled_idx[self.settings.train_size + self.settings.validate_size:]
        return MatrixObservedData(
            row_features,
            col_features,
            train_indices,
            validate_indices,
            test_indices,
            observed_matrix,
            alpha,
            beta,
            gamma,
            true_matrix
        )

    def matrix_completion_groups(self, sv_val=1, feat_vec_factor=1):
        matrix_shape = (self.settings.num_rows, self.settings.num_cols)

        def _make_feature_vec(num_feat, num_nonzero_groups, num_total_groups):
            return (
                [feat_vec_factor * np.matrix(np.ones(num_feat)).T] * num_nonzero_groups
                + [np.matrix(np.zeros(num_feat)).T] * (num_total_groups - num_nonzero_groups)
            )

        def _create_feature_matrix(num_samples, num_feat):
            return np.matrix(np.random.randn(num_samples, num_feat))

        alphas = _make_feature_vec(
            self.settings.num_row_features,
            self.settings.num_nonzero_row_groups,
            self.settings.num_row_groups
        )
        betas = _make_feature_vec(
            self.settings.num_row_features,
            self.settings.num_nonzero_row_groups,
            self.settings.num_row_groups
        )

        row_features = [
            _create_feature_matrix(self.settings.num_rows, self.settings.num_row_features)
            for i in range(self.settings.num_row_groups)
        ]
        col_features = [
            _create_feature_matrix(self.settings.num_cols, self.settings.num_col_features)
            for i in range(self.settings.num_col_groups)
        ]

        gamma = 0
        for i in range(self.settings.num_nonzero_s):
            u = np.random.multivariate_normal(np.zeros(self.settings.num_rows), np.eye(self.settings.num_rows))
            v = np.random.multivariate_normal(np.zeros(self.settings.num_cols), np.eye(self.settings.num_cols))
            gamma += sv_val * np.matrix(u).T * np.matrix(v)

        true_matrix = get_matrix_completion_groups_fitted_values(
            row_features,
            col_features,
            alphas,
            betas,
            gamma,
        )

        epsilon = np.random.randn(matrix_shape[0], matrix_shape[1])
        SNR_factor = self._make_snr_factor(np.linalg.norm(true_matrix, ord="fro"), np.linalg.norm(epsilon))
        observed_matrix = true_matrix + 1.0 / SNR_factor * epsilon

        # index column-major style
        shuffled_idx = np.random.permutation(matrix_shape[0] * matrix_shape[1])
        train_indices = shuffled_idx[0:self.settings.train_size]
        validate_indices = shuffled_idx[self.settings.train_size:self.settings.train_size + self.settings.validate_size]
        test_indices = shuffled_idx[self.settings.train_size + self.settings.validate_size:]
        return MatrixGroupsObservedData(
            row_features,
            col_features,
            train_indices,
            validate_indices,
            test_indices,
            observed_matrix,
            alphas,
            betas,
            gamma,
            true_matrix
        )

    def _make_data(self, true_y, observed_X):
        # Given the true y and corresponding observed X values, this will add noise so that the SNR is correct
        epsilon = np.matrix(np.random.randn(self.total_samples, 1))
        SNR_factor = self._make_snr_factor(np.linalg.norm(true_y), np.linalg.norm(epsilon))
        observed_y = true_y + 1.0 / SNR_factor * epsilon

        X_train, X_validate, X_test = self._split_data(observed_X)
        y_train, y_validate, y_test = self._split_y_vector(observed_y)

        return ObservedData(X_train, y_train, X_validate, y_validate, X_test, y_test)

    def _split_y_vector(self, y):
        return y[0:self.train_size], y[self.train_size:self.train_size + self.validate_size], y[self.train_size + self.validate_size:]

    def _split_data(self, X):
        return X[0:self.train_size, :], X[self.train_size:self.train_size + self.validate_size, :], X[self.train_size + self.validate_size:, :]

    def _make_shuffled_uniform_X(self, eps=0.0001):
        step_size = (self.feat_range[1] - self.feat_range[0] + eps)/self.total_samples
        # start the uniformly spaced X at a different start point, jitter by about 1/20 of the step size
        jitter = np.random.uniform(0, 1) * step_size/10
        equal_spaced_X = np.arange(self.feat_range[0] + jitter, self.feat_range[1] + jitter, step_size)
        np.random.shuffle(equal_spaced_X)
        return equal_spaced_X

    def _make_snr_factor(self, true_sig_norm, noise_norm):
        return self.snr/ true_sig_norm * noise_norm
