"""
Implementation of the Policy Iteration algorithm given page 8 of 
"Adaptive Aggregation Methods for Infinite Horizon 
Dynamic Programming" - Bertsekas, Castanon.
"""

import numpy as np
import time

from utils.generic_model import GenericModel, NUMPY, SPARSE
from utils.partition_fixed_value import FixedValuePartition
from utils.calculus import (
    bellman_operator,
    bellman_policy_operator,
    iterative_policy_evaluation,
    compute_transition_reward_policy,
)


def inv_approximate(matrix: np.ndarray, tolerance=1e-1):
    res = np.zeros((matrix.shape[0], matrix.shape[0]))
    x = np.eye(matrix.shape[0]) - matrix
    step = 0
    while True:
        val = x**step
        res += val
        if np.linalg.norm(val) < tolerance:
            break
        step += 1
    return res


class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
        beta_1: float = 0.00001,
        beta_2: float = 0.001,
        fixed_number_of_regions: int = 40,
        step_approx_y: int = 1000,
        verbose: bool = False,
        mode: str = SPARSE,
    ):
        # Class arguments
        self.model = model
        self.discount = discount
        self.epsilon_policy_evaluation = final_precision
        self.epsilon_final_policy_evaluation = final_precision
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.fixed_number_of_regions = fixed_number_of_regions
        self.step_approx_y = step_approx_y
        self.verbose = verbose
        self.mode = mode

        self.model._convert_model(self.mode)
        self.name = "Bertsekas"

        self.fixed_partition = FixedValuePartition(
            self.model, self.discount, fixed_number_of_regions
        )

    def span_state_space(self, value: np.ndarray) -> float:
        return value.max() - value.min()

    def run(self):
        start_time = time.time()
        self.policy = np.zeros((self.model.state_dim))

        while True:
            # Policy Evaluation
            self.value = self._policy_evaluation(
                self.policy,
            )

            # Policy Update
            new_policy = bellman_operator(self.model, self.value, self.discount).argmax(
                axis=1
            )

            policy_condition = np.all(new_policy == self.policy)

            if policy_condition:
                if self.verbose:
                    print("Optimal Policy Reached.")
                transition_policy, reward_policy = self._compute_transition_reward_pi(
                    self.policy
                )

                self.value = iterative_policy_evaluation(
                    transition_policy,
                    reward_policy,
                    self.discount,
                    self.epsilon_final_policy_evaluation,
                )

                self.runtime = time.time() - start_time

                break
            else:
                self.policy = new_policy

    def _policy_evaluation(self, policy: np.ndarray) -> np.ndarray:
        transition_policy, reward_policy = compute_transition_reward_policy(
            self.model, policy
        )
        self.omega_1 = np.inf
        self.omega_2 = np.inf
        self.value = np.zeros((self.model.state_dim))

        while True:
            bellman_value = bellman_policy_operator(
                self.value, self.discount, transition_policy, reward_policy
            )
            span_tv_minus_v = self.span_state_space(bellman_value - self.value)

            if span_tv_minus_v < self.epsilon_policy_evaluation:
                self.value += self.discount / 2 / (1 - self.discount) * span_tv_minus_v
                return self.value
            else:
                if span_tv_minus_v <= self.omega_1 and span_tv_minus_v >= self.omega_2:
                    self.omega_1 = self.beta_1 * span_tv_minus_v
                else:
                    self.omega_2 = self.beta_2 * span_tv_minus_v
                    self.value = bellman_value
                    continue

                self.fixed_partition.build_partition_along_value(
                    bellman_value - self.value
                )
                self.fixed_partition._build_weights(recompute=True)
                Q = self.fixed_partition.weights
                self.fixed_partition._compute_weights_phi()
                W = self.fixed_partition._partial_phi()
                right_y = Q @ (bellman_value - self.value)

                A = np.eye(self.fixed_partition.region_number)
                prod = Q @ transition_policy @ W
                self.fixed_partition._build_weights(recompute=True)
                left_y_before_inv = A - self.discount * prod

                left_y = inv_approximate(left_y_before_inv)

                y = left_y @ right_y
                value_1 = self.value + W @ y
                self.value = value_1
                self.omega_2 = np.inf

    def _compute_transition_reward_pi(self, policy):
        if not isinstance(self.model.transition_matrix, list):
            transition_policy = np.empty((self.model.state_dim, self.model.state_dim))
            reward_policy = np.zeros(self.model.state_dim)
            for aa in range(self.model.action_dim):
                ind = (policy == aa).nonzero()[0]
                # if no rows use action a, then no need to assign this
                if ind.size > 0:
                    transition_policy[ind, :] = self.model.transition_matrix[aa][ind, :]
                    reward_policy[ind] = self.model.reward_matrix[ind, aa]

            return transition_policy, reward_policy
        else:
            transition_policy = np.empty((self.model.state_dim, self.model.state_dim))
            reward_policy = np.zeros(self.model.state_dim)
            for aa in range(self.model.action_dim):
                ind = (policy == aa).nonzero()[0]
                # if no rows use action a, then no need to assign this
                if ind.size > 0:
                    transition_policy[ind, :] = self.model.transition_matrix[aa][
                        ind, :
                    ].toarray()
                    reward_policy[ind] = self.model.reward_matrix[ind, aa]

            return transition_policy, reward_policy
