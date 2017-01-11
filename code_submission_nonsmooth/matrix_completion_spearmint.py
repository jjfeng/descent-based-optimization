from spearmint_algo import Spearmint_Algo
from common import testerror_matrix_completion
from convexopt_solvers import MatrixCompletionProblemWrapperCustom
from convexopt_solvers import MatrixCompletionProblemWrapperSimple

class Matrix_Completion_Spearmint(Spearmint_Algo):
    method_label = "Matrix_Completion_Spearmint"
    result_folder_prefix = "spearmint_descent/matrix_completion"

    def _create_problem_wrapper(self):
        self.problem_wrapper = MatrixCompletionProblemWrapperCustom(
            self.data
        )
        self.num_lambdas = 5

    def get_validation_cost(self, model_params):
        return testerror_matrix_completion(
            self.data,
            self.data.validate_idx,
            model_params
        )

class Matrix_Completion_Spearmint_Simple(Spearmint_Algo):
    method_label = "Matrix_Completion_Spearmint_Simple"
    result_folder_prefix = "spearmint_descent/matrix_completion"

    def _create_problem_wrapper(self):
        self.problem_wrapper = MatrixCompletionProblemWrapperSimple(
            self.data
        )
        self.num_lambdas = 2

    def get_validation_cost(self, model_params):
        return testerror_grouped(
            self.data,
            self.data.validate_idx,
            model_params
        )
