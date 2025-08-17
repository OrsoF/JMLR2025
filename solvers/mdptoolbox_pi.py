"""
Solver calling the MDP Toolbox Value Iteration solver.
"""

from mdptoolbox.mdp import PolicyIteration
import numpy as np
from utils.generic_model import GenericModel
from utils.generic_solver import GenericSolver
import time


class Solver(GenericSolver):
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
    ):
        self.model = model
        self.discount = discount
        self.epsilon = final_precision
        self.name = "VImdptoolbox"
        self.max_iter = int(1e8)

        self.value: np.ndarray
        self.policy: np.ndarray

    def run(self):
        start_time = time.time()

        self.vi = PolicyIteration(
            self.model.transition_matrix,
            self.model.reward_matrix,
            discount=self.discount,
            max_iter=self.max_iter,
            skip_check=True,
        )
        self.vi.run()
        self.runtime = time.time() - start_time

        self.value = np.array(self.vi.V)
        self.policy = np.array(self.vi.policy)
