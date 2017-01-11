from grid_search import Grid_Search
from common import testerror_matrix_completion
from convexopt_solvers import MatrixCompletionProblemWrapperSimple

class Matrix_Completion_Grid_Search(Grid_Search):
    method_label = "Matrix_Completion_Grid_Search"

    def _create_problem_wrapper(self):
        self.problem_wrapper = MatrixCompletionProblemWrapperSimple(self.data)

    def get_validation_cost(self, model_params):
        return testerror_matrix_completion(
            self.data,
            self.data.validate_idx,
            model_params
        )
