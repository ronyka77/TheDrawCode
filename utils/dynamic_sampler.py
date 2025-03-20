import optuna
from optuna.samplers import TPESampler
from optuna.distributions import UniformDistribution, IntUniformDistribution

class DynamicTPESampler(TPESampler):
    """
    A custom TPESampler that adjusts the search space for specific parameters
    if the trial number exceeds a threshold.
    
    Args:
        dynamic_threshold (int): After this many trials, use the dynamic search space.
        dynamic_search_space (dict): A mapping from parameter names to a function which,
            given the original distribution, returns a new distribution.
    """
    def __init__(self, dynamic_threshold=200, dynamic_search_space=None, **kwargs):
        self.dynamic_threshold = dynamic_threshold
        self.dynamic_search_space = dynamic_search_space or {}
        super().__init__(**kwargs)

    def sample_independent(self, study, trial, param_name, param_distribution):
        if trial.number >= self.dynamic_threshold and param_name in self.dynamic_search_space:
            new_distribution = self.dynamic_search_space[param_name](param_distribution)
            return super().sample_independent(study, trial, param_name, new_distribution)
        return super().sample_independent(study, trial, param_name, param_distribution)