from utils.generic_model import GenericModel
from typing import Tuple
from utils.partition_generic import GenericPartition


class ValuePartition(GenericPartition):
    def __init__(self, model: GenericModel, discount: float, mode: str):
        self.model = model
        self.discount = discount
        self.mode = mode

        self._initialize_trivial_aggregation()

    def _compute_aggregate_transition_reward(
        self,
    ) -> Tuple:
        """
        Get the aggregate transition function w @ T @ phi and the aggregate reward function w @ R.
        """
        if not hasattr(self, "aggregate_transition_matrix") and not hasattr(
            self, "aggregate_reward_matrix"
        ):
            self._compute_weights_phi()
            # We build it if None.
            partial_phi = self._partial_phi()
            # self._build_weights()

            self.aggregate_transition_matrix = [
                self.model.transition_matrix[aa].dot(partial_phi)
                for aa in range(self.model.action_dim)
            ]

            self.aggregate_reward_matrix = self.model.reward_matrix

            return self.aggregate_transition_matrix, self.aggregate_reward_matrix

        elif hasattr(self, "aggregate_transition_matrix") and hasattr(
            self, "aggregate_reward_matrix"
        ):
            # Already built, return it.
            return self.aggregate_transition_matrix, self.aggregate_reward_matrix
        else:
            assert False, "This case should not be seen."
