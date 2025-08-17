"""
Solver calling the MDP Toolbox Modified Policy Iteration solver.
"""

from mdptoolbox.mdp import PolicyIterationModified
import numpy as np
from utils.generic_model import GenericModel
import time


class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
    ):
        self.model = model
        self.epsilon = final_precision
        self.max_step_policy_evaluation = int(1e5)

        self.name = "PIMmdptoolbox"
        self.value: np.ndarray
        self.policy: np.ndarray

    def run(self):
        start_time = time.time()

        self.pim = PolicyIterationModified(
            self.model.transition_matrix,
            self.model.reward_matrix,
            discount=1.0,
            epsilon=self.epsilon,
            max_iter=self.max_step_policy_evaluation,
            skip_check=True,
        )
        self.pim.run()
        self.runtime = time.time() - start_time

        self.value = np.array(self.pim.V)
        self.policy = np.array(self.pim.policy)
