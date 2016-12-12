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

    def _make_data(self, true_y, observed_X):
        # Given the true y and corresponding observed X values, this will add noise so that the SNR is correct
        epsilon = np.matrix(np.random.randn(self.total_samples, 1))
        SNR_factor = self.snr / np.linalg.norm(true_y) * np.linalg.norm(epsilon)
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
