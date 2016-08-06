from common import testerror_sparse_add_smooth
from neldermead import Nelder_Mead_Algo
from convexopt_solvers import SparseAdditiveModelProblemWrapper

class Sparse_Add_Model_Nelder_Mead(Nelder_Mead_Algo):
    method_label = "Sparse_Add_Model_Nelder_Mead"
    MAX_COST = 100000

    def _create_problem_wrapper(self):
        self.problem_wrapper = SparseAdditiveModelProblemWrapper(
            self.data.X_full,
            self.data.train_idx,
            self.data.y_train
        )

    def get_validation_cost(self, lambdas):
        # if any are not positive, then just return max value
        for l in lambdas:
            if l <= 0:
                return self.MAX_COST

        thetas = self.problem_wrapper.solve(lambdas, quick_run=True)
        validation_cost = testerror_sparse_add_smooth(
            self.data.y_validate,
            self.data.validate_idx,
            thetas
        )
        self.log("validation_cost %f" % validation_cost)
        return validation_cost
