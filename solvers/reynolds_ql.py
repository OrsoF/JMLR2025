"""
Implementation of Q-Value Iteration.
Here, the Q-value is updated by blocks
following an iteratively refined partition.
"""

import re
from tracemalloc import start
import numpy as np
from utils.exact_value_function import distance_to_optimal
from utils.generic_model import GenericModel
from utils.generic_solver import GenericSolver
from time import time
from utils.partition_generic import GenericPartition as QValuePartition
from utils.calculus import (
    generate_trajectory,
    norminf,
    q_optimal_bellman_operator,
    generate_sars_random_trajectory_epsilon_greedy,
    generate_trajectory_partition,
    optimal_bellman_residual,
)
from utils.calculus_projected import (
    apply_poqbo_until_var_small,
    projected_optimal_q_bellman_operator,
)
from tqdm import trange, tqdm
from params import (
    traj_count,
    traj_length,
    learning_rate,
    best_action,
    decay_rate,
    replay_count,
)

NUMPY, SPARSE = "numpy", "sparse"
rng = np.random.default_rng(0)


class Solver(GenericSolver):
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
    ):
        # Class arguments
        self.model = model
        self.discount = discount
        self.epsilon = final_precision
        self.model._convert_model(NUMPY)
        self.model._normalize_reward_matrix()
        self.name = "Reynolds"
        self.partition = QValuePartition(self.model, self.discount)
        self.steps_done_learning = 0

        self.significant_loss_delta_min = 0.001

        self.infos = {
            "error_to_optimal": [],
            "number_of_regions": [],
        }

    def alpha(self) -> float:
        self.steps_done_learning += 1
        return 0.25 * learning_rate / self.steps_done_learning**decay_rate

    def contracted_q_learning_traj(self, contracted_q_value: np.ndarray) -> np.ndarray:
        value = self.partition._partial_phi().dot(contracted_q_value).max(axis=1)
        trajectory = generate_trajectory_partition(
            self.model, self.partition, traj_length, best_action, value
        )
        for _ in range(replay_count):
            for elem in trajectory:
                region, action, reward, next_region = elem
                delta = (
                    reward
                    + self.discount * contracted_q_value[next_region].max()
                    - contracted_q_value[region, action]
                )
                contracted_q_value[region, action] += (
                    self.alpha() * delta / len(self.partition.states_in_region[region])
                )
        return contracted_q_value

    def q_learning_traj(self, q_value: np.ndarray) -> np.ndarray:
        trajectory = generate_trajectory(self.model, traj_length, best_action, q_value)
        for _ in range(replay_count):
            for elem in trajectory:
                state, action, reward, next_state = elem
                delta = (
                    reward
                    + self.discount * q_value[next_state].max()
                    - q_value[state, action]
                )
                q_value[state, action] += self.alpha() * delta
        return q_value

    def run(self):
        start_time = time()
        self.infos["error_to_optimal"] = []
        self.infos["number_of_regions"] = []

        contracted_q_value = np.zeros(
            (self.partition._number_of_regions(), self.model.action_dim)
        )
        for _ in range(traj_count):
            if self.partition._number_of_regions() == self.model.state_dim:
                contracted_q_value = self.q_learning_traj(contracted_q_value)
                value = contracted_q_value.max(axis=1)
                error = optimal_bellman_residual(self.model, value, self.discount)
                self.infos["error_to_optimal"].append(error)
            else:
                self.partition._compute_phi()  # For the error calculation
                contracted_q_value = self.contracted_q_learning_traj(contracted_q_value)

                value = (
                    self.partition._partial_phi().dot(contracted_q_value).max(axis=1)
                )
                error = distance_to_optimal(
                    value,
                    self.model,
                    self.discount,
                )
                self.infos["error_to_optimal"].append(error)

                self.region_division_step(contracted_q_value)
                print(
                    "Number of regions after division:",
                    self.partition._number_of_regions(),
                )
                self.infos["number_of_regions"].append(
                    self.partition._number_of_regions()
                )
                contracted_q_value = self.update_contracted_q_value(contracted_q_value)

        self.runtime = time() - start_time
        self.value = self.partition._partial_phi().dot(contracted_q_value).max(axis=1)

    def update_contracted_q_value(self, contracted_q_value: np.ndarray) -> np.ndarray:
        if contracted_q_value.shape[0] == self.partition._number_of_regions():
            return contracted_q_value
        else:
            new_contracted_q_value = np.zeros(
                (self.partition._number_of_regions(), self.model.action_dim)
            )
            new_contracted_q_value[: contracted_q_value.shape[0]] = contracted_q_value
            return new_contracted_q_value

    def find_adjacent_regions_sampling(
        self, region_index: int, traj_len: int = 10, exp_number: int = 10
    ) -> list:
        adjacent_regions = [region_index]
        if self.partition._number_of_regions() == 1:
            return adjacent_regions

        for _ in range(exp_number):
            state = rng.choice(self.partition.states_in_region[region_index])
            for _ in range(traj_len):
                state_region = self.partition.get_region_index(state)
                if state_region != region_index:
                    adjacent_regions.append(state_region)
                    break
                action = rng.integers(self.model.action_dim)
                state = rng.choice(
                    self.model.state_dim, p=self.model.transition_matrix[action, state]
                )
        return adjacent_regions

    def find_adjacent_regions(self, region_index: int) -> list:
        """Compute the list of regions that are adjacent to the region_index."""
        self.model._model_to_sparse()
        self.partition.reset_attributes()
        self.partition._compute_weights_phi()
        self.partition.compute_agg_trans_reward_q()
        aggregated_transition = self.partition.aggregate_transition_matrix
        connected_regions = []
        for region_index_2 in range(self.partition._number_of_regions()):
            transition_to_this_region = [
                aggregated_transition[aa][region_index, region_index_2]
                for aa in range(self.model.action_dim)
            ]
            if max(transition_to_this_region) > 0.0:
                connected_regions.append(region_index_2)

        self.model._model_to_numpy()
        return connected_regions

    def visit_condition(self, region_index: int, action: int) -> bool:
        return True

    def divide_region(self, region_index: int):
        current_region = self.partition.states_in_region[region_index]
        if len(current_region) <= 1:
            return
        else:
            self.partition.states_in_region[region_index] = current_region[
                : len(current_region) // 2
            ]
            self.partition.states_in_region.append(
                current_region[len(current_region) // 2 :]
            )

    def region_division_step(self, contracted_q_value: np.ndarray):
        for region_index in range(self.partition._number_of_regions()):
            connected_region_indices = self.find_adjacent_regions_sampling(region_index)
            for region_index_2 in connected_region_indices:
                aa1 = contracted_q_value[region_index].argmax()
                aa2 = contracted_q_value[region_index_2].argmax()
                action_condition = aa1 != aa2

                delta_1 = abs(
                    contracted_q_value[region_index, aa1]
                    - contracted_q_value[region_index, aa2]
                )
                delta_2 = abs(
                    contracted_q_value[region_index_2, aa1]
                    - contracted_q_value[region_index_2, aa2]
                )
                significant_loss_condition = (
                    delta_1 >= self.significant_loss_delta_min
                    or delta_2 >= self.significant_loss_delta_min
                )

                all_regions_visit_condition = max(
                    [
                        self.visit_condition(region_ind, action)
                        for region_ind in [region_index, region_index_2]
                        for action in range(self.model.action_dim)
                    ]
                )

                if (
                    action_condition
                    and significant_loss_condition
                    and all_regions_visit_condition
                    or rng.random() < 1.0
                ):
                    self.divide_region(region_index)
                    self.divide_region(region_index_2)
                    contracted_q_value = self.update_contracted_q_value(
                        contracted_q_value
                    )
