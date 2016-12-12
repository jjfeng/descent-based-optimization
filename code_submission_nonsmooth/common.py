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

def betaerror(beta_real, beta_guess):
    return np.linalg.norm(beta_real - beta_guess)

def get_nonzero_indices(some_vector, threshold=CLOSE_TO_ZERO_THRESHOLD):
    return np.reshape(np.array(np.greater(np.abs(some_vector), threshold).T), (some_vector.size, ))

def get_norm2(vector, power=1):
    return np.power(np.linalg.norm(vector, ord=None), power)
