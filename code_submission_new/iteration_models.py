class Simulation_Settings:
    train_size = 100
    validate_size = 50
    test_size = 50
    snr = 2
    gs_num_lambdas = 10
    spearmint_numruns = 100
    nm_iters = 50
    feat_range = [-5,5]
    method = "HC"
    plot = False

class Iteration_Data:
    def __init__(self, i, data, settings):
        self.data = data
        self.settings = settings
        self.i = i
