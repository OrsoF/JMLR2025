"""
Implementation of modified Policy Iteration
with a changed Policy Evaluation step.

The new Policy Evaluation successively slices
the state space to update by block the current
value function V^pi.
"""

from typing import Generic
from matplotlib import axis
import numpy as np
from utils.calculus_partition import span_by_region
from utils.generic_model import GenericModel, NUMPY, SPARSE
import time
from utils.partition_generic import GenericPartition as PiPartition
from utils.calculus import (
    norminf,
    bellman_operator,
    value_span_on_regions,
    bellman_policy_operator,
    compute_transition_reward_policy,
    optimal_bellman_operator,
    iterative_policy_evaluation,
)
from utils.calculus_projected import (
    apply_ppbo_until_var_small,
    projected_policy_bellman_operator,
)
from utils.calculus import bellman_residual_policy

def q_policy_bellman(q_value: np.ndarray, policy: np.ndarray, discount: float, model:GenericModel) -> np.ndarray:
    return model.reward_matrix + discount * model.transition_matrix.dot(q_value[policy])

def agg_q_policy_bellman(
    agg_transition: np.ndarray,
    agg_reward: np.ndarray,
    contracted_q_value: np.ndarray,
    discount: float,
    policy: np.ndarray,
) -> np.ndarray:
    return agg_reward + discount * agg_transition.dot(contracted_q_value[policy])

class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
        verbose: bool = False,
    ):
        # Class arguments
        self.model = model
        self.discount = discount
        self.epsilon_final_policy_evaluation = final_precision
        self.epsilon_variation = final_precision
        self.epsilon_policy_evaluation = final_precision
        self.verbose = verbose

        assert self.discount < 1.0, "Use aggregated_pim_total instead."

        self.proj_bellman_max_steps = np.inf

        self.model._convert_model(NUMPY)
        self.name = "PDQPIM"
        self.partition = PiPartition(self.model, self.discount)

        self.state_dim = self.model.state_dim
        self.action_dim = self.model.action_dim

    def _disagg_policy_evaluation(
        self,
        policy: np.ndarray,
        epsilon_policy_evaluation: float,
        contracted_q_value: np.ndarray,
    ) -> np.ndarray:
        epsilon_pbr = (1 - self.discount) * epsilon_policy_evaluation / 2
        epsilon_span = (1 - self.discount) * epsilon_policy_evaluation / 2
        # self.partition.reset_attributes()

        self.partition._compute_weights_phi()

        transition_policy, reward_policy = compute_transition_reward_policy(
            self.model, policy
        )

        while True:
            # Common values
            agg_reward_policy, agg_transition_policy = self.partition.compute_agg_trans_reward_pi(
                transition_policy, reward_policy
            )
            agg_reward = self.partition.weights.dot(self.model.reward_matrix)
            agg_transition = self.partition.weights.dot(self.model.transition_matrix.dot(self.partition._partial_phi()))

            # Span condition
            q_value = self.partition._partial_phi().dot(contracted_q_value)
            exact_policy_bellman = self.model.reward_matrix + self.discount * self.model.transition_matrix.dot(q_value[policy])
            span_condition = (
                np.all([max(span_by_region(value, self.partition.states_in_region)) < epsilon_span for value in exact_policy_bellman.T])
            )
            print(f"Span condition: {span_condition}, epsilon_span: {epsilon_span}")

            # Precision condition
            agg_bellman_value = agg_q_policy_bellman(agg_transition, agg_reward, contracted_q_value, self.discount, policy)
            precision_condition = (
                norminf(agg_bellman_value - contracted_q_value) < epsilon_pbr
            )
            print(f"Precision condition: {precision_condition}, epsilon_pbr: {epsilon_pbr}")

            if span_condition and precision_condition:
                return q_value

            while norminf(agg_bellman_value - contracted_q_value) >= epsilon_pbr:
                contracted_q_value = agg_bellman_value
                agg_bellman_value = agg_reward + self.discount * agg_transition.dot(
                    contracted_q_value
                )

            q_value = self.partition._partial_phi().dot(contracted_q_value)
            exact_policy_bellman = (
                reward_policy + self.discount * transition_policy.dot(q_value)
            )
            span_condition = (
                np.max(exact_policy_bellman) - np.min(exact_policy_bellman)
                < epsilon_span
            )

            if not span_condition:
                for value in q_value.T:
                    contracted_q_value = self.partition.divide_all_regions_along_value_update_contracted_q_value(
                        value, epsilon_span, contracted_q_value
                    )

                self.partition._compute_weights_phi()

    def run(self):
        start_time = time.time()

        policy = np.zeros((self.state_dim), dtype=int)
        contracted_q_value = np.zeros(
            (self.partition._number_of_regions(), self.action_dim)
        )

        while True:
            # Policy Evaluation
            contracted_q_value = self._disagg_policy_evaluation(
                policy, self.epsilon_policy_evaluation, contracted_q_value
            )

            assert False

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
                # print(
                #     "Number of policy updates - AggPIM : {}".format(
                #         number_policy_update
                #     )
                # )
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
        epsilon_pbr = (1 - self.discount) * epsilon_policy_evaluation / 2
        epsilon_span = (1 - self.discount) * epsilon_policy_evaluation / 2
        # self.partition.reset_attributes()

        self.partition._compute_weights_phi()
        contracted_value = self.partition.weights.dot(initial_full_value)

        transition_policy, reward_policy = compute_transition_reward_policy(
            self.model, policy
        )

        # self.partition.update_transition_reward_policy(transition_policy, reward_policy)

        while True:
            if self.partition._number_of_regions() > 0.9 * self.model.state_dim:
                self.partition._compute_weights_phi()
                value = self.partition._partial_phi().dot(contracted_value)
                value = iterative_policy_evaluation(
                    transition_policy,
                    reward_policy,
                    self.discount,
                    epsilon_policy_evaluation,
                    value,
                )
                return value

            # self.partition._compute_aggregate_transition_reward_policy(True)
            self.partition._compute_weights_phi()
            self.partition.compute_agg_trans_reward_pi(transition_policy, reward_policy)

            if self.partition._number_of_regions() > 1:
                contracted_value, pbr_value = apply_ppbo_until_var_small(
                    self.discount,
                    self.partition.aggregate_transition_policy,
                    self.partition.aggregate_reward_policy,
                    epsilon_pbr,
                    contracted_value,
                    self.proj_bellman_max_steps,
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
            contracted_value = (
                self.partition.divide_all_regions_along_value_update_contracted_value(
                    bellman_of_value, epsilon_span, contracted_value
                )
            )
