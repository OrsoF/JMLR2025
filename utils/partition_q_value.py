from utils.generic_model import GenericModel
import numpy as np
from scipy.sparse import lil_matrix, lil_matrix
import time
from utils.calculus_partition import (
    get_weight_matrix_from_states_in_regions,
    get_full_phi_from_states_in_regions,
)
from utils.calculus_projected import projected_optimal_q_bellman_operator
from typing import Tuple, List
from utils.partition_generic import GenericPartition

NUMPY, SPARSE = "numpy", "sparse"


class QValuePartition(GenericPartition):
    """
    We assume a partition is described by
    """

    def __init__(self, model: GenericModel, discount: float, mode: str) -> None:
        # Parsing arguments
        self.model = model
        self.discount = discount
        self.mode = mode

        self.states_in_region = [list(range(self.model.state_dim))]
        if discount >= 1.0:
            region_0 = []
            region_1 = []
            for ss in range(self.model.state_dim):
                transition = min(
                    self.model.transition_matrix[aa][ss, ss]
                    for aa in range(self.model.action_dim)
                )
                if transition >= 1.0:
                    region_0.append(ss)
                else:
                    region_1.append(ss)
            if len(region_0):
                self.states_in_region = [region_0, region_1]
            else:
                self.states_in_region = [region_1]

    def projected_q_bellman_operator(self, contracted_q_value):
        """
        Get (Pi T_Q)(contracted_q_value)
        """
        self._compute_aggregate_transition_reward()

        return projected_optimal_q_bellman_operator(
            self.model,
            self.discount,
            contracted_q_value,
            self.aggregate_transition_matrix,
            self.aggregate_reward_matrix,
        )

    def _compute_aggregate_transition_reward(self):
        """
        Get the aggregate transition function w @ T @ phi and the aggregate reward function w @ R.
        """
        if not hasattr(self, "aggregate_transition_matrix") and not hasattr(
            self, "aggregate_reward_matrix"
        ):
            # We build it if None.
            self._compute_weights_phi()
            partial_phi = self._partial_phi()

            self.aggregate_transition_matrix = [
                self.weights.dot(self.model.transition_matrix[aa].dot(partial_phi))
                for aa in range(self.model.action_dim)
            ]

            self.aggregate_reward_matrix = self.weights @ self.model.reward_matrix

            return self.aggregate_transition_matrix, self.aggregate_reward_matrix

        elif hasattr(self, "aggregate_transition_matrix") and hasattr(
            self, "aggregate_reward_matrix"
        ):
            # Already built, return it.
            return self.aggregate_transition_matrix, self.aggregate_reward_matrix
        else:
            assert False, "This case should not be seen."
