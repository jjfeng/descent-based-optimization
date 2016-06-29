import math
import numpy as np

from common import *

# Y is a linear function, where some covariates are correlated
def correlated(train_size, num_features, num_nonzero_features, signal_noise_ratio=1):
    validate_size = _get_validate_size(train_size)
    total_samples = train_size + validate_size + TEST_SIZE

    correlation_matrix = np.matrix([[math.pow(0.5, abs(i - j)) for i in range(0, num_features)] for j in range(0, num_features)])
    X = np.matrix(np.random.randn(total_samples, num_features)) * np.matrix(np.linalg.cholesky(correlation_matrix)).T

    # beta real is a shuffled array of zeros and iid std normal values
    beta_real = np.matrix(
        np.concatenate((
            np.ones((num_nonzero_features, 1)), # np.random.randn(num_nonzero_features, 1),
            np.zeros((num_features - num_nonzero_features, 1))
        ))
    )

    np.random.shuffle(beta_real)

    # add gaussian noise
    epsilon = np.matrix(np.random.randn(total_samples, 1))

    # Get the appropriate signal noise ratio
    SNR_factor = signal_noise_ratio / np.linalg.norm(X * beta_real) * np.linalg.norm(epsilon)
    y = X * beta_real + (1.0 / SNR_factor) * epsilon

    return _return_split_dataset(X, y, train_size, validate_size, beta_real)

# Y is a linear function of the covariates, where three groups of variables are very correlated to each other
def three_groups(train_size, num_features, num_nonzero_features, signal_noise_ratio=1):
    NOISE_STD = 0.1
    COEFF = 3

    validate_size = _get_validate_size(train_size)
    total_samples = train_size + validate_size + TEST_SIZE

    # X1, X2, X3 are the groups of features that are *highly* correlated
    size = int(num_nonzero_features/3)
    X1 = _get_tiled_matrix(total_samples, size) + np.random.randn(total_samples, size) * NOISE_STD
    X2 = _get_tiled_matrix(total_samples, size) + np.random.randn(total_samples, size) * NOISE_STD

    remaining_nonzero_features = num_nonzero_features - 2 * size
    X3 = _get_tiled_matrix(total_samples, remaining_nonzero_features) + np.random.randn(total_samples, remaining_nonzero_features) * NOISE_STD
    X4 = np.random.randn(total_samples, num_features - num_nonzero_features)
    X = np.matrix(np.hstack((X1, X2, X3, X4)))

    beta_real = np.matrix(
        np.concatenate((
            COEFF * np.ones((num_nonzero_features, 1)),
            np.zeros((num_features - num_nonzero_features, 1))
        ))
    )

    # Get the appropriate signal noise ratio
    beta_real = _get_rescaled_beta(signal_noise_ratio, X, beta_real)

    # add gaussian noise
    epsilon = np.matrix(np.random.randn(total_samples, 1))

    y = X * beta_real + epsilon

    return _return_split_dataset(X, y, train_size, validate_size, beta_real)

# Y is a linear function of the covaraites, but the coefficients are very sparse
def sparse_groups(train_size, group_feature_sizes, validate_size_ratio=3, desired_signal_noise_ratio=2):
    BASE_NONZERO_COEFF = [1, 2, 3, 4, 5]
    NONZERO_FEATURES = len(BASE_NONZERO_COEFF)

    validate_size = train_size/validate_size_ratio #_get_validate_size(train_size)
    total_samples = train_size + validate_size + TEST_SIZE

    X = np.matrix(np.random.randn(total_samples, np.sum(group_feature_sizes)))
    betas = [np.matrix(np.concatenate((BASE_NONZERO_COEFF, [0 for i in range(0, num_features - NONZERO_FEATURES)]))).T for num_features in group_feature_sizes]
    beta = np.matrix(np.concatenate(betas))

    epsilon = np.matrix(np.random.randn(total_samples, 1))

    SNR_factor = 1.0 / desired_signal_noise_ratio / np.linalg.norm(epsilon) * np.linalg.norm(X * beta)
    y = X * beta + SNR_factor * epsilon

    return _return_split_dataset(X, y, train_size, validate_size, betas)

# Y is the sum of a linear function and smooth function
def smooth_plus_linear(train_size, num_features, num_nonzero_features, data_type=0, linear_to_smooth_ratio=1, desired_signal_noise_ratio=4):
    NOISE_STD = 1.0/16
    MIN_X_SMOOTH = 0.0
    MAX_X_SMOOTH = 1.0
    NUM_NONZERO_FEATURE_GROUPS = 2

    validate_size = _get_validate_size(train_size)
    total_samples = train_size + validate_size + TEST_SIZE

    # Generate the linear covariates
    nonzero_feature_group_size = num_nonzero_features / NUM_NONZERO_FEATURE_GROUPS
    X1 = _get_tiled_matrix(total_samples, nonzero_feature_group_size) + np.random.randn(total_samples, nonzero_feature_group_size) * NOISE_STD
    X1 += np.random.randn(total_samples, nonzero_feature_group_size) * NOISE_STD

    X2 = _get_tiled_matrix(total_samples, nonzero_feature_group_size) + np.random.randn(total_samples, nonzero_feature_group_size) * NOISE_STD
    X2 += np.random.randn(total_samples, nonzero_feature_group_size) * NOISE_STD

    X3 = np.random.randn(total_samples, num_features - nonzero_feature_group_size * NUM_NONZERO_FEATURE_GROUPS)
    X_linear = np.matrix(np.hstack((X1, X2, X3)))

    # True beta for the linear component
    beta_real = np.matrix(
        np.concatenate((
            np.ones((num_nonzero_features, 1)) * 0.2,
            np.zeros((num_features - num_nonzero_features, 1))
        ))
    )

    # Generate the smooth covariates
    X_smooth = np.matrix(np.random.uniform(MIN_X_SMOOTH, MAX_X_SMOOTH, total_samples)).T

    # Generate the smooth component
    if data_type == 0:
        y_smooth = np.sin(X_smooth * 5) + np.sin(15 * (X_smooth - 3))
    elif data_type == 1:
        y_smooth = np.matrix(np.multiply(MAX_X_SMOOTH + 1 - X_smooth, np.sin(20 * np.power(X_smooth, 4))))
    elif data_type == 2:
        y_smooth = 4 * np.power(X_smooth, 3) - 4 * np.power(X_smooth, 2) + X_smooth

    # Scale the smooth component so we get the requested linear to smooth ratio
    y_smooth *= 1.0 / linear_to_smooth_ratio * np.linalg.norm(X_linear * beta_real) / np.linalg.norm(y_smooth)

    # Generate noise and scale it for the requested signal to noise ratio
    epsilon = np.matrix(np.random.randn(total_samples, 1))
    SNR_factor = desired_signal_noise_ratio / np.linalg.norm(X_linear * beta_real + y_smooth) * np.linalg.norm(epsilon)

    # Combine all the data
    y = X_linear * beta_real + y_smooth + 1.0 / SNR_factor * epsilon

    # Separate data into train, validate, and test components
    Xl_train, Xl_validate, Xl_test = _split_data(X_linear, train_size, validate_size)
    Xs_train, Xs_validate, Xs_test = _split_data(X_smooth, train_size, validate_size)
    y_train, y_validate, y_test = _split_y_vector(y, train_size, validate_size)
    y_smooth_train, y_smooth_validate, y_smooth_test = _split_y_vector(y_smooth, train_size, validate_size)

    return beta_real, Xl_train, Xs_train, y_train, Xl_validate, Xs_validate, y_validate, Xl_test, Xs_test, y_test, y_smooth_train, y_smooth_validate, y_smooth_test

# @param smooth_fcn_list: the smooth functions for generating Y
def multi_smooth_features(train_size, smooth_fcn_list, desired_snr=2, feat_range=[0,1], train_to_validate_ratio=15, test_size=20):
    def make_X_smooth_feature():
        step_size = (feat_range[1] - feat_range[0] + 0.0001)/total_samples
        jitter = np.random.uniform(0, 1) * step_size/10
        equal_spaced_X = np.arange(feat_range[0] + jitter, feat_range[1] + jitter, step_size)
        np.random.shuffle(equal_spaced_X)
        return equal_spaced_X

    validate_size = np.floor(train_size/train_to_validate_ratio)
    total_samples = train_size + validate_size + test_size
    num_features = len(smooth_fcn_list)

    X_smooth = make_X_smooth_feature()
    for i in range(num_features - 1):
        X_smooth = np.column_stack((X_smooth, make_X_smooth_feature()))
    y_smooth = 0
    for idx, fcn in enumerate(smooth_fcn_list):
        y_smooth += fcn(X_smooth[:, idx]).reshape(total_samples, 1)

    epsilon = np.matrix(np.random.randn(total_samples, 1))
    SNR_factor = desired_snr / np.linalg.norm(y_smooth) * np.linalg.norm(epsilon)
    y = y_smooth + 1.0 / SNR_factor * epsilon

    X_train, X_validate, X_test = _split_data(X_smooth, train_size, validate_size)
    y_train, y_validate, y_test = _split_y_vector(y, train_size, validate_size)

    return X_train, y_train, X_validate, y_validate, X_test, y_test

# Helper functions
def _return_split_dataset(X, y, train_size, validate_size, beta_real):
    X_train, X_validate, X_test = _split_data(X, train_size, validate_size)
    y_train, y_validate, y_test = _split_y_vector(y, train_size, validate_size)

    return beta_real, X_train, y_train, X_validate, y_validate, X_test, y_test

def _split_y_vector(y, train_size, validate_size):
    return y[0:train_size], y[train_size:train_size + validate_size], y[train_size + validate_size:]

def _split_data(X, train_size, validate_size):
    return X[0:train_size, :], X[train_size:train_size + validate_size, :], X[train_size + validate_size:, :]

# Returns a new beta that is rescaled to have the correct signal noise ratio
def _get_rescaled_beta(desired_signal_noise_ratio, X, beta, epsilon):
    return beta * desired_signal_noise_ratio / np.linalg.norm(X * beta) * np.linalg.norm(epsilon)

def _get_validate_size(train_size):
    return int(train_size / TRAIN_TO_VALIDATE_RATIO)

def _get_tiled_matrix(total_samples, length):
    return np.tile(np.random.randn(total_samples, 1), (1, length))
