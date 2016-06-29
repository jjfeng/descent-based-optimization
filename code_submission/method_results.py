import numpy as np

class MethodResults:
    def __init__(self, method_name):
        self.method_name = method_name
        self.beta_errs = []
        self.test_errs = []
        self.theta_errs = []
        self.sensitivities = []
        self.validation_errs = []
        self.runtimes = []

    def get_num_runs(self):
        return len(self.test_errs)

    def print_results(self):
        print self.method_name, "Results: (mean, std dev)"
        print "beta: %.4f, %.4f" % (np.average(self.beta_errs), np.var(self.beta_errs))
        print "validation: %.4f, %.4f" % (np.average(self.validation_errs), np.var(self.validation_errs))
        print "test: %.4f, %.4f" % (np.average(self.test_errs), np.var(self.test_errs))

        if len(self.theta_errs):
            print "theta: %.4f, %.4f" % (np.average(self.theta_errs), np.var(self.theta_errs))
        if len(self.sensitivities):
            print "sensitivity: %.4f, %.4f" % (np.average(self.sensitivities), np.var(self.sensitivities))

        print "runtimes: %.4f, %.4f" % (np.average(self.runtimes), np.var(self.runtimes))

    def append(self, result):
        if result.test_err is not None:
            self.test_errs.append(result.test_err)
        else:
            raise Exception("test error missing")

        if result.beta_err is not None:
            self.beta_errs.append(result.beta_err)

        if result.validation_err is not None:
            self.validation_errs.append(result.validation_err)
        else:
            raise Exception("validation error missing")

        if result.runtime is not None:
            self.runtimes.append(result.runtime)
        else:
            raise Exception("runtime missing")

        if result.theta_err is not None:
            self.theta_errs.append(result.theta_err)

        if result.sensitivity is not None:
            self.sensitivities.append(result.sensitivity)


    def append_test_beta_err(self, beta_test_errs):
        self.test_errs.append(beta_test_errs[0])
        self.beta_errs.append(beta_test_errs[1])

    def append_test_beta_theta_err(self, errs):
        self.test_errs.append(errs[0])
        self.beta_errs.append(errs[1])
        self.theta_errs.append(errs[2])

    def append_runtime(self, runtime):
        self.runtimes.append(runtime)

    def append_validation_err(self, validation_err):
        self.validation_errs.append(validation_err)

class MethodResult:
    def __init__(self, test_err=None, beta_err=None, validation_err=None, theta_err=None, sensitivity=None, runtime=None):
        self.test_err = test_err
        self.beta_err = beta_err
        self.validation_err = validation_err
        self.theta_err = theta_err
        self.runtime = runtime
        self.sensitivity = sensitivity
