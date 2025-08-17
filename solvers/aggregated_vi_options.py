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
        mode: str = SPARSE,
        verbose: bool = False,
        bellman_updates: int = 10,
        decreasing_span_bound: bool = False,
    ):
        # Class arguments
        self.model = model
        self.discount = discount
        self.epsilon = final_precision
        self.verbose = verbose
        self.bellman_updates = bellman_updates
        self.mode = mode
        self.decreasing_span_bound = float(decreasing_span_bound)

        self.model._convert_model(self.mode)
        self.name = "PDVIoptions"

        # Variables
        self.partition = ValuePartition(self.model, self.discount, self.mode)
        self.contracted_value: np.ndarray

        self.epsilon_pbr = (
            self.epsilon * (1 - self.discount) / 2
        )  # Bound on the projected bellman residual
        self.epsilon_span = (
            self.epsilon * (1 - self.discount) / 2
        )  # Bound on the span of bellman(V)

    def compute_weights(self, norm_method: str) -> np.ndarray:
        if norm_method == "region_size":
            norm_weights = 1 / self.partition._partial_phi().sum(axis=0).squeeze()
        elif norm_method == "average_reward":
            norm_weights = np.array(
                [
                    self.model.reward_matrix[region, :].max(axis=1).mean() + 1e-2
                    for region in self.partition.states_in_region
                ]
            )
        elif norm_method == "average_average_reward":
            norm_weights = np.array(
                [
                    self.model.reward_matrix[region, :].mean() + 1e-2
                    for region in self.partition.states_in_region
                ]
            )
        elif norm_method == "max_reward_region":
            norm_weights = np.array(
                [
                    self.model.reward_matrix[region, :].max()
                    for region in self.partition.states_in_region
                ]
            )
        return norm_weights

    def run(
        self,
        with_norm_weights: bool = True,
        norm_method: str = "max_reward_region",
        max_agg_step: int = int(1e10),
    ):
        start_time = time()
        self.contracted_value = np.zeros((len(self.partition.states_in_region)))
        self.steps = 0

        while True:
            self.partition._compute_weights_phi()
            self.partition.compute_agg_trans_reward_v()

            if with_norm_weights:
                norm_weights = self.compute_weights(norm_method)
            else:
                norm_weights = None

            self.contracted_value, pbr_value = apply_pobo_until_var_small(
                self.model,
                self.discount,
                self.partition.aggregate_transition_matrix,
                self.partition.aggregate_reward_matrix,
                self.partition.weights,
                self.epsilon_pbr,
                self.contracted_value,
                self.mode,
                norm_weights,
            )

            # Once we sufficiently applied PB operator
            # We compute maximum span to eventually change partition
            bellman_of_value = self._exact_bellman_step(self.contracted_value)
            maximum_span = max(
                span_by_region(bellman_of_value, self.partition.states_in_region)
            )

            if self.verbose:
                print("Maximum span : {}".format(maximum_span))
                print("K : {}".format(len(self.partition.states_in_region)))

            if maximum_span > self.epsilon_span:
                # If maximum span is greater than the wanted bound

                epsi_pbr = (
                    self._epsilon_span() * self.decreasing_span_bound
                    + self.epsilon_span * (1 - self.decreasing_span_bound)
                )
                self.contracted_value = self.partition.divide_all_regions_along_value_update_contracted_value(
                    bellman_of_value,
                    epsi_pbr,
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
                    self.mode,
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

    def _epsilon_span(self) -> float:
        if not hasattr(self, "const"):
            self.const = 1
        else:
            self.const += 1
        return self.epsilon_span + 1000 * self.epsilon_span / self.const**5
