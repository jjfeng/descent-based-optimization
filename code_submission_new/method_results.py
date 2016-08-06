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
        self.lambda_sets = []

    def get_num_runs(self):
        return len(self.test_errs)

    def print_results(self):
        num_runs = len(self.validation_errs)
        def get_std_err(values):
            return np.sqrt(np.var(values)/num_runs)

        if len(self.validation_errs) > 0:
            print self.method_name, "Results: (mean, std dev)"
            print "validation: %.4f, %.4f" % (np.average(self.validation_errs), get_std_err(self.validation_errs))
            print "test: %.4f, %.4f" % (np.average(self.test_errs), get_std_err(self.test_errs))
            if len(self.beta_errs):
                print "beta: %.4f, %.4f" % (np.average(self.beta_errs), get_std_err(self.beta_errs))
            if len(self.theta_errs):
                print "theta: %.4f, %.4f" % (np.average(self.theta_errs), get_std_err(self.theta_errs))
            if len(self.sensitivities):
                print "sensitivity: %.4f, %.4f" % (np.average(self.sensitivities), get_std_err(self.sensitivities))

            print "runtimes: %.4f, %.4f" % (np.average(self.runtimes), get_std_err(self.runtimes))
            if len(self.lambda_sets):
                print "average lambdas: %s" % np.mean(np.vstack(self.lambda_sets), axis=0)

    def append(self, result):
        if result.validation_err is None:
            # ignore input if no validation error. something went wrong with this run
            return

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

        if result.lambdas is not None:
            self.lambda_sets.append(result.lambdas)

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
    def __init__(self, test_err=None, beta_err=None, validation_err=None, theta_err=None, sensitivity=None, runtime=None, lambdas=None):
        self.test_err = test_err
        self.beta_err = beta_err
        self.validation_err = validation_err
        self.theta_err = theta_err
        self.runtime = runtime
        self.sensitivity = sensitivity
        self.lambdas = lambdas

    def __str__(self):
        return_str = "testerr %f\nvalidation_err %f\n" % (self.test_err, self.validation_err)
        if self.beta_err is not None:
            return_str += "beta_err %f\n" % self.beta_err
        if self.theta_err is not None:
            return_str += "theta_err %f\n" % self.theta_err
        if self.sensitivity is not None:
            return_str += "sensitivity %f\n" % self.sensitivity
        if self.lambdas is not None:
            return_str += "lambdas %s\n" % self.lambdas
        return_str += "runtime %f\n" % self.runtime
        return return_str
