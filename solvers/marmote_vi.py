"""
Solver calling the Marmote Value Iteration solver in C++.
"""

from marmote.core import MarmoteInterval, SparseMatrix, FullMatrix
from marmote.mdp import DiscountedMDP, SolutionMDP
from time import time
import numpy as np
from utils.generic_model import GenericModel
from utils.numpy_to_marmote import (
    build_marmote_transition_list,
    build_marmote_reward_matrix,
)


class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
    ):
        self.model = model
        self.discount = discount
        self.epsilon = final_precision
        self.max_iter = int(1e8)
        self.name = "VImarmote"

    def run(self):
        self.state_space = MarmoteInterval(0, int(self.model.state_dim - 1))
        self.action_space = MarmoteInterval(0, int(self.model.action_dim - 1))

        self.reward_matrix = build_marmote_reward_matrix(
            self.model.state_dim, self.model.action_dim, self.model.reward_matrix
        )

        self.transitions_list = build_marmote_transition_list(
            self.model.state_dim, self.model.action_dim, self.model.transition_matrix
        )

        self.mdp = DiscountedMDP(
            "max",
            self.state_space,
            self.action_space,
            self.transitions_list,
            self.reward_matrix,
            self.discount,
        )

        self.start_time = time()

        self.opt: SolutionMDP = self.mdp.ValueIteration(self.epsilon, self.max_iter)

        self.runtime = time() - self.start_time

        self.value = np.array(
            [self.opt.getValueIndex(ss) for ss in range(self.model.state_dim)]
        )
        self.policy = None
