from spearmint_algo import Spearmint_Algo
from common import testerror_grouped
from convexopt_solvers import GroupedLassoProblemWrapper

class SGL_Spearmint(Spearmint_Algo):
    method_label = "SGL_Spearmint"
    result_folder_prefix = "spearmint_descent/sgl"

    def _create_problem_wrapper(self):
        self.problem_wrapper = GroupedLassoProblemWrapper(
            self.data.X_train,
            self.data.y_train,
            self.settings.get_expert_group_sizes()
        )
        self.num_lambdas = self.settings.expert_num_groups + 1

    def get_validation_cost(self, model_params):
        return testerror_grouped(
            self.data.X_validate,
            self.data.y_validate,
            model_params
        )
