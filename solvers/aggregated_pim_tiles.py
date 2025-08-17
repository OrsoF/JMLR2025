"""
Implementation of modified Policy Iteration 
with a changed Policy Evaluation step.

The new Policy Evaluation successively slices 
the state space to update by block the current 
value function V^pi.
"""

import numpy as np
from utils.generic_model import GenericModel, SPARSE
import time
from utils.partition_generic import GenericPartition as PiPartition
from utils.calculus import (
    norminf,
    bellman_operator,
    value_span_on_regions,
    bellman_policy_operator,
    compute_transition_reward_policy,
    optimal_bellman_operator,
)
from utils.calculus_projected import (
    apply_ppbo_until_var_small,
)


class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
        verbose: bool = False,
        bellman_updates: int = 1,
        n_tiles: int = 10,
    ):
        # Class arguments
        self.model = model
        self.discount = discount
        self.epsilon_final_policy_evaluation = final_precision
        self.epsilon_variation = final_precision
        self.epsilon_policy_evaluation = final_precision
        self.verbose = verbose
        self.bellman_updates = bellman_updates
        self.n_tiles = n_tiles
        self.n_tiles = int((model.state_dim) ** (1 / 2)) + 1

        self.model._convert_model(SPARSE)
        self.name = "PDPIMtiles"
        self.partition = PiPartition(self.model, self.discount)

    def run(self):
        start_time = time.time()
        self.policy = np.zeros((self.model.state_dim))
        self.value = np.zeros((self.model.state_dim))
        self.contracted_value = np.zeros((self.partition._number_of_regions()))

        while True:
            # Policy Evaluation
            self.value = self._policy_evaluation(
                self.policy,
                self.epsilon_policy_evaluation,
                self.value,
            )

            # Policy Update
            q_value = bellman_operator(self.model, self.value, self.discount)
            new_policy, new_value = q_value.argmax(axis=1), q_value.max(axis=1)

            condition_variation = norminf(
                new_value - self.value
            ) < self.epsilon_variation * (1 - self.discount)
            condition_policy = np.all(new_policy == self.policy)

            self.value = new_value

            if condition_variation or condition_policy:
                self.runtime = time.time() - start_time
                break

            self.policy = new_policy

    def _get_maximum_span_and_bellman_value(
        self,
        contracted_value: np.ndarray,
        transition_policy: np.ndarray,
        reward_policy: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Returns Span(R + gamma.T @ phi @ contracted_V)
        """
        partial_phi = self.partition._partial_phi()
        value = partial_phi @ contracted_value
        bellman_of_value = bellman_policy_operator(
            value, self.discount, transition_policy, reward_policy
        )
        return (
            max(
                value_span_on_regions(bellman_of_value, self.partition.states_in_region)
            ),
            bellman_of_value,
        )

    def _policy_evaluation(
        self,
        policy: np.ndarray,
        epsilon_policy_evaluation: float,
        initial_full_value: np.ndarray,
    ) -> np.ndarray:
        # self.partition.reset_attributes()
        epsilon_pbr = (1 - self.discount) * epsilon_policy_evaluation / 2
        epsilon_span = (1 - self.discount) * epsilon_policy_evaluation / 2

        self.partition._compute_weights_phi()
        contracted_value = self.partition.weights.dot(initial_full_value)

        transition_policy, reward_policy = compute_transition_reward_policy(
            self.model, policy
        )

        # self.partition.update_transition_reward_policy(transition_policy, reward_policy)
        while True:
            # self.partition._compute_aggregate_transition_reward_policy(True)
            self.partition._compute_weights_phi()
            self.partition.compute_agg_trans_reward_pi(transition_policy, reward_policy)

            contracted_value, pbr_value = apply_ppbo_until_var_small(
                self.discount,
                self.partition.aggregate_transition_policy,
                self.partition.aggregate_reward_policy,
                epsilon_pbr,
                contracted_value,
            )

            # Compute span
            maximum_span, bellman_of_value = self._get_maximum_span_and_bellman_value(
                contracted_value, transition_policy, reward_policy
            )

            for _ in range(self.bellman_updates):
                bellman_of_value = optimal_bellman_operator(
                    self.model, bellman_of_value, self.discount
                )

            maximum_span = max(
                value_span_on_regions(bellman_of_value, self.partition.states_in_region)
            )

            self.partition._compute_weights_phi()
            self.partition.compute_agg_trans_reward_pi(transition_policy, reward_policy)

            # If span and pbr small, break
            if maximum_span < epsilon_span and pbr_value < epsilon_pbr:
                self.value = self.partition._partial_phi() @ contracted_value
                return self.value

            # Else, the span should be big (normally), we divide partition
            # contracted_value = self.partition.divide_regions_along_tv(
            #     bellman_of_value,
            #     epsilon_span,
            #     contracted_value,
            # )
            contracted_value = self.partition.divide_all_regions_along_value_update_contracted_value_tiles(
                bellman_of_value, contracted_value, self.n_tiles
            )
