import os
import numpy as np

TRAIN_TO_VALIDATE_RATIO = 4
TEST_SIZE = 200

# verbosity of convex optimization solver
VERBOSE = False

# method "X" means it is the many-parameter version
# method "X0" is the simple 2-parameter version
METHODS = ["NM", "NM0", "HC", "HC0", "GS", "SP", "SP0"]

X_CORR = 0
W_CORR = 0.9

CLOSE_TO_ZERO_THRESHOLD = 1e-4

def testerror(X, y, b):
    return 0.5 * get_norm2(y - X * b, power=2)

def testerror_grouped(X, y, betas):
    complete_beta = np.concatenate(betas)
    diff = y - X * complete_beta
    return 0.5 / y.size * get_norm2(diff, power=2)

def testerror_logistic_grouped(X, y, betas):
    complete_beta = np.concatenate(betas)

    # get classification rate
    probability = np.power(1 + np.exp(-1 * X * complete_beta), -1)
    print "guesses", np.hstack([probability, y])

    num_correct = 0
    for i, p in enumerate(probability):
        if y[i] == 1 and p >= 0.5:
            num_correct += 1
        elif y[i] <= 0 and p < 0.5:
            num_correct += 1

    correct_classification_rate = float(num_correct) / y.size
    print "correct_classification_rate", correct_classification_rate

    # get loss value
    Xb = X * complete_beta
    log_likelihood = -1 * y.T * Xb + np.sum(np.log(1 + np.exp(Xb)))

    return log_likelihood, correct_classification_rate

def testerror_elastic_net(X, y, b):
    return 0.5/y.size * get_norm2(y - X * b, power=2)

def testerror_sparse_add_smooth(y, test_indices, thetas):
    err = y - np.sum(thetas[test_indices, :], axis=1)
    return 0.5/y.size * get_norm2(err, power=2)

def testerror_matrix_completion(data, indices, model_params):
    fitted_m = get_matrix_completion_fitted_values(
        data.row_features,
        data.col_features,
        model_params["row_theta"],
        model_params["col_theta"],
        model_params["interaction_m"]
    )
    # index column-major style
    return 0.5/indices.size * get_norm2(make_column_major_flat(data.observed_matrix - fitted_m)[indices], power=2)

def make_column_major_flat(m):
    return np.reshape(m, (m.size, 1), order='F')

def make_column_major_reshape(m, shape):
    return np.reshape(m, shape, order='F')

def get_matrix_completion_fitted_values(row_feat, col_feat, row_theta, col_theta, interaction_m):
    num_rows = row_feat.shape[0]
    num_cols = col_feat.shape[0]
    row_component = row_feat * row_theta * np.ones(num_rows).T
    col_component = (col_feat * col_theta * np.ones(num_cols).T).T
    return row_component + col_component + interaction_m

def betaerror(beta_real, beta_guess):
    return np.linalg.norm(beta_real - beta_guess)

def get_nonzero_indices(some_vector, threshold=CLOSE_TO_ZERO_THRESHOLD):
    return np.reshape(np.array(np.greater(np.abs(some_vector), threshold).T), (some_vector.size, ))

def get_norm2(vector, power=1):
    return np.power(np.linalg.norm(vector, ord=None), power)

def get_intersection_percent(idx1, denom_idx2):
    s1 = np.array(idx1)
    s2 = np.array(denom_idx2)
    if s2.size == 0:
        return 100.0
    return np.intersect1d(s1, s2).size * 100.0/ s2.size
