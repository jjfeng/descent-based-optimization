import numpy as np

TRAIN_TO_VALIDATE_RATIO = 4
TEST_SIZE = 200

# verbosity of convex optimization solver
VERBOSE = False

HC_LAMBDA12_LABEL = "HillclimbLambda12"
HC_LAMBDA12_COLOR = "red"
HC_LAMBDA12_DIM_LABEL = "HillclimbLambda12Dim"
HC_LAMBDA12_DIM_COLOR = "purple"
HC_LAMBDA12_NESTEROV_LABEL = "HillclimbLambda12Nesterov"
HC_LAMBDA12_NESTEROV_COLOR = "blue"
HC_LAMBDA_ALPHA_LABEL = "HillclimbLambdaAlpha"
HC_LAMBDA_ALPHA_COLOR = "green"
HC_LAMBDA_ALPHA_DIM_LABEL = "HillclimbLambdaAlphaDim"
HC_LAMBDA_ALPHA_DIM_COLOR = "orange"
HC_LAMBDA_ALPHA_NESTEROV_LABEL = "HillclimbLambdaAlphaNesterov"
HC_LAMBDA_ALPHA_NESTEROV_COLOR = "gray"
GS_LAMBDA12_LABEL = "GridsearchLambda12"
GS_COLOR = "brown"

HC_GROUPED_LASSO_LABEL = "HillclimbGroupedLasso"
HC_GROUPED_LASSO_COLOR = "red"
GS_GROUPED_LASSO_LABEL = "GridsearchGroupedLasso"

EFFECTS_TO_INTERACTION_RATIO = 5
EFFECTS_TO_OTHER_RATIO = 5
X_CORR = 0
W_CORR = 0.9

CLOSE_TO_ZERO_THRESHOLD = 1e-4

def testerror(X, y, b):
    return 0.5 * get_norm2(y - X * b, power=2)

def testerror_interactions(X, W, y, b, t):
    return 0.5 * get_norm2(y - X * b - W * t, power=2)

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

def testerror_smooth_and_linear(X_linear, y, beta, thetas):
    return 0.5 * get_norm2(y - X_linear * beta - thetas, power=2)

def testerror_multi_smooth(y, test_indices, thetas):
    err = y - np.sum(thetas[test_indices], axis=1)
    return 0.5/y.size * get_norm2(err, power=2)

def betaerror(beta_real, beta_guess):
    return np.linalg.norm(beta_real - beta_guess)

def get_nonzero_indices(some_vector, threshold=CLOSE_TO_ZERO_THRESHOLD):
    return np.reshape(np.array(np.greater(np.abs(some_vector), threshold).T), (some_vector.size, ))

def get_norm2(vector, power=1):
    return np.power(np.linalg.norm(vector, ord=None), power)

#### RANDOM TESTS

assert(get_norm2(np.array([2,2,2,2])) == 4)
assert(get_norm2(np.array([2,2,2,2]), 3) == 64)
