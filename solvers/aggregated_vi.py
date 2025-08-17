"""
This code allows to slice a state space progressively 
along an evolving value function.
"""

import numpy as np
from utils.generic_model import GenericModel, SPARSE
from time import time
from utils.partition_generic import GenericPartition as ValuePartition
from utils.calculus import apply_obo_until_var_small, optimal_bellman_operator
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
        verbose: bool = False,
        bellman_updates: int = 10,
        proj_bellman_max_steps: int = np.inf,
    ):
        # Class arguments
        self.model = model
        self.discount = discount
        self.epsilon = final_precision
        self.verbose = verbose
        self.bellman_updates = bellman_updates
        self.proj_bellman_max_steps = proj_bellman_max_steps

        # self.model._convert_model(SPARSE)
        self.name = "PDVI"

        # Variables
        self.partition = ValuePartition(self.model, self.discount)
        self.contracted_value: np.ndarray

        self.proj_bellman_max_steps = max(
            100,
            10
            ** (
                int(np.log10(min(self.model.state_dim, int(1 / (1 - self.discount)))))
                - 1
            ),
        )

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
            if self.partition._number_of_regions() > 0.9 * self.model.state_dim:
                self.value = self.partition._partial_phi().dot(self.contracted_value)
                self.value = apply_obo_until_var_small(
                    self.model, self.discount, 2 * self.epsilon_pbr, self.value
                )
                self.runtime = time() - start_time
                return

            self.partition._compute_weights_phi()
            self.partition.compute_agg_trans_reward_v()

            if self.partition._number_of_regions() > 1:
                self.contracted_value, pbr_value = apply_pobo_until_var_small(
                    self.model,
                    self.discount,
                    self.partition.aggregate_transition_matrix,
                    self.partition.aggregate_reward_matrix,
                    self.partition.weights,
                    self.epsilon_pbr * 100,
                    self.contracted_value,
                    max_steps=self.proj_bellman_max_steps,
                )

            # Once we sufficiently applied PB operator
            # We compute maximum span to eventually change partition
            bellman_of_value = self._exact_bellman_step(self.contracted_value)
            self.contracted_value = self.partition.weights.dot(bellman_of_value)
            maximum_span = max(
                span_by_region(bellman_of_value, self.partition.states_in_region)
            )

            if maximum_span > self.epsilon_span:
                # If maximum span is greater than the wanted bound
                self.contracted_value = self.partition.divide_all_regions_along_value_update_contracted_value(
                    bellman_of_value,
                    self.epsilon_span,
                    self.contracted_value,
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
