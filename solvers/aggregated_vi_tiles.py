"""
This code allows to slice a state space progressively 
along an evolving value function.
"""

import numpy as np
from utils.generic_model import GenericModel, SPARSE
from time import time
from utils.partition_generic import GenericPartition as ValuePartition
from utils.calculus import optimal_bellman_operator
from utils.calculus_projected import (
    apply_pobo_until_var_small,
    projected_optimal_bellman_operator_residual,
)
from utils.calculus_partition import span_by_region


class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
        bellman_updates: int = 1,
        n_tiles: int = 10,
    ):
        # Class arguments
        self.model = model
        self.discount = discount
        self.epsilon = final_precision
        self.bellman_updates = bellman_updates
        self.n_tiles = n_tiles
        self.n_tiles = int((model.state_dim) ** (1 / 4))

        self.model._convert_model(SPARSE)
        self.name = "PDVItiles"

        # Variables
        self.partition = ValuePartition(self.model, self.discount)
        self.contracted_value: np.ndarray

        self.epsilon_pbr = (
            self.epsilon * (1 - self.discount) / 2
        )  # Bound on the projected bellman residual
        self.epsilon_span = (
            self.epsilon * (1 - self.discount) / 2
        )  # Bound on the span of bellman(V)

    def run(
        self,
    ):
        start_time = time()
        self.contracted_value = np.zeros((len(self.partition.states_in_region)))
        self.steps = 0

        while True:
            self.partition._compute_weights_phi()
            self.partition.compute_agg_trans_reward_v()

            self.contracted_value, pbr_value = apply_pobo_until_var_small(
                self.model,
                self.discount,
                self.partition.aggregate_transition_matrix,
                self.partition.aggregate_reward_matrix,
                self.partition.weights,
                self.epsilon_pbr,
                self.contracted_value,
            )

            # Once we sufficiently applied PB operator
            # We compute maximum span to eventually change partition
            bellman_of_value = self._exact_bellman_step(self.contracted_value)
            maximum_span = max(
                span_by_region(bellman_of_value, self.partition.states_in_region)
            )

            if maximum_span > self.epsilon_span:
                # If maximum span is greater than the wanted bound

                self.contracted_value = self.partition.divide_all_regions_along_value_update_contracted_value_tiles(
                    bellman_of_value,
                    self.contracted_value,
                    self.n_tiles,
                )

                # self.partition._compute_aggregate_transition_reward()
                self.partition._compute_weights_phi()
                self.partition.compute_agg_trans_reward_v()

                pbr_value = projected_optimal_bellman_operator_residual(
                    self.model,
                    self.discount,
                    self.contracted_value,
                    self.partition.aggregate_transition_matrix,
                    self.partition.aggregate_reward_matrix,
                    self.partition.weights,
                )

            if maximum_span < self.epsilon_span and pbr_value < self.epsilon_pbr:
                self.runtime = time() - start_time
                self.value = self.partition._partial_phi().dot(self.contracted_value)
                break

    def _exact_bellman_step(self, contracted_value: np.ndarray) -> np.ndarray:
        value = self.partition._partial_phi().dot(contracted_value)

        bellman_of_value = optimal_bellman_operator(self.model, value, self.discount)
        for _ in range(self.bellman_updates):
            bellman_of_value = optimal_bellman_operator(
                self.model, bellman_of_value, self.discount
            )

        return bellman_of_value

    def _maximum_span_bellman(self, contracted_value: np.ndarray) -> float:
        bellman_value = self._exact_bellman_step(contracted_value)
        spans = span_by_region(bellman_value, self.partition.states_in_region)
        return max(spans)
