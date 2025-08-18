"""
Implementation of Q-Value Iteration. 
Here, the Q-value is updated by blocks 
following an iteratively refined partition.
"""

import numpy as np
from utils.generic_model import GenericModel
from utils.generic_solver import GenericSolver
from time import time
from utils.partition_generic import GenericPartition as QValuePartition
from utils.calculus import (
    apply_obo_until_var_small,
    norminf,
    q_optimal_bellman_operator,
)
from utils.calculus_projected import (
    apply_poqbo_until_var_small,
    projected_optimal_q_bellman_operator,
)

NUMPY, SPARSE = "numpy", "sparse"


class Solver(GenericSolver):
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
        verbose: bool = False,
        bellman_updates: int = 100,
    ):
        # Class arguments
        self.model = model
        self.discount = 1.0
        self.epsilon = final_precision
        self.verbose = verbose
        self.bellman_updates = bellman_updates
        self.proj_bellman_max_steps = 100
        self.name = "PDQVI"
        self.model._convert_model(SPARSE)

        # Variables
        self.partition = QValuePartition(self.model, self.discount)
        self.contracted_q_value: np.ndarray

        self.epsilon_pbr = self.epsilon / 2
        self.epsilon_span = self.epsilon / 2

    def run(self):
        start_time = time()
        self.contracted_q_value = np.zeros(
            (len(self.partition.states_in_region), self.model.action_dim)
        )

        while True:
            if self.partition._number_of_regions() > 0.9 * self.model.state_dim:
                self.value = self.partition._partial_phi().dot(
                    self.contracted_q_value.max(axis=1)
                )
                self.value, _ = apply_obo_until_var_small(
                    self.model,
                    self.discount,
                    self.epsilon,
                    self.value,
                )
                self.runtime = time() - start_time

                return

            self.partition._compute_weights_phi()
            self.partition.compute_agg_trans_reward_q()

            if self.partition._number_of_regions() > 1:
                self.contracted_q_value, pbr_value = apply_poqbo_until_var_small(
                    self.model,
                    self.discount,
                    self.partition.aggregate_transition_matrix,
                    self.partition.aggregate_reward_matrix,
                    self.epsilon_pbr,
                    self.contracted_q_value,
                    max_steps=self.proj_bellman_max_steps,
                )
            # else:
            #     T = np.array(
            #         [elem[0, 0] for elem in self.partition.aggregate_transition_matrix]
            #     )
            #     R = self.partition.aggregate_reward_matrix
            #     q = np.max(np.divide(R, 1 - self.discount * T))
            #     self.contracted_q_value = R + self.discount * q * T

            q_value = self.partition._partial_phi().dot(self.contracted_q_value)

            bellman_of_q_value = q_optimal_bellman_operator(
                self.model, q_value, self.discount
            )

            for _ in range(self.bellman_updates):
                bellman_of_q_value = q_optimal_bellman_operator(
                    self.model, bellman_of_q_value, self.discount
                )

            maximum_span = np.max(
                self._get_spans_q_value_on_each_region(bellman_of_q_value)
            )

            if maximum_span > self.epsilon_span:
                # If maximum span is greater than the wanted bound

                for action in range(self.model.action_dim):
                    self.contracted_q_value = self.partition.divide_all_regions_along_value_update_contracted_q_value(
                        bellman_of_q_value[:, action],
                        self.epsilon_span,
                        self.contracted_q_value,
                    )

                self.partition._compute_weights_phi()
                self.partition.compute_agg_trans_reward_q()
            projected_q_bellman_value = projected_optimal_q_bellman_operator(
                self.model,
                self.discount,
                self.contracted_q_value,
                self.partition.aggregate_transition_matrix,
                self.partition.aggregate_reward_matrix,
            )
            pbr_value = norminf(projected_q_bellman_value - self.contracted_q_value)

            if maximum_span < self.epsilon_span and pbr_value < self.epsilon_pbr:
                self.runtime = time() - start_time
                self.value = self.partition._partial_phi().dot(
                    self.contracted_q_value.max(axis=1)
                )
                break

    def _get_spans_q_value_on_each_region(
        self, full_q_value_function: np.ndarray
    ) -> list:
        """
        Return a list :
        [max(value[region])-min(value[region]) for region in regions]
        """
        return [
            np.ptp(full_q_value_function[region, :], axis=0)
            for region in self.partition.states_in_region
        ]
