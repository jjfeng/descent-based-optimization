from common import testerror_matrix_completion
from neldermead import Nelder_Mead_Algo
from convexopt_solvers import MatrixCompletionProblemWrapperCustom
from convexopt_solvers import MatrixCompletionProblemWrapperSimple

class Matrix_Completion_Nelder_Mead(Nelder_Mead_Algo):
    method_label = "Matrix_Completion_Nelder_Mead"
    MAX_COST = 100000

    def _create_problem_wrapper(self):
        self.problem_wrapper = MatrixCompletionProblemWrapperCustom(self.data)

    def get_validation_cost(self, lambdas):
        # if any are not positive, then just return max value
        for l in lambdas:
            if l <= 0:
                return self.MAX_COST

        model_params = self.problem_wrapper.solve(lambdas, quick_run=True)
        validation_cost = testerror_matrix_completion(
            self.data,
            self.data.validate_idx,
            model_params
        )
        self.log("validation_cost %f" % validation_cost)
        return validation_cost


class Matrix_Completion_Nelder_Mead_Simple(Nelder_Mead_Algo):
    method_label = "Matrix_Completion_Nelder_Mead_simple"
    MAX_COST = 100000

    def _create_problem_wrapper(self):
        self.problem_wrapper = MatrixCompletionProblemWrapperStupid(self.data)

    def get_validation_cost(self, lambdas):
        # if any are not positive, then just return max value
        for l in lambdas:
            if l <= 0:
                return self.MAX_COST

        model_params = self.problem_wrapper.solve(lambdas, quick_run=True)
        validation_cost = testerror_matrix_completion(
            self.data,
            self.data.validate_idx,
            model_params
        )
        self.log("validation_cost %f" % validation_cost)
        return validation_cost
