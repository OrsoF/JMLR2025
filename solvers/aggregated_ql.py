from cProfile import label
from calendar import c
from os import error
import re
from turtle import rt
from matplotlib import axis
import numpy as np
import time
from regex import F, P
from sympy import comp
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns

from utils.generic_model import GenericModel, NUMPY, SPARSE
from utils.partition_q_value import QValuePartition, GenericPartition
from utils.calculus import (
    generate_sars,
    generate_trajectory,
    generate_trajectory_partition,
    norminf,
    optimal_bellman_residual,
    convert_trajectory_to_region,
    optimal_bellman_operator,
    q_optimal_bellman_operator,
    is_policy_optimal,
)
from utils.exact_value_function import (
    get_exact_value,
    compute_exact_value,
    distance_to_optimal,
)
from utils.model_conversion import ModelToGym
from utils.generic_solver import GenericSolver
from params import (
    traj_count,
    traj_length,
    learning_rate,
    decay_rate,
    best_action,
    replay_count,
    br_learning_rate,
    epsilon_division,
    sample_before_division,
)

from utils.data_management import solve

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
        self.model._model_to_numpy()

        self.discount = discount
        self.name = "Aggregated Q-Learning"
        self.steps_done_learning = 0

    def alpha(self) -> float:
        self.steps_done_learning += 1
        return learning_rate / self.steps_done_learning**decay_rate

    def ql_update(
        self, qv: np.ndarray, trajectory: list, replay_count: int
    ) -> np.ndarray:
        for _ in range(replay_count):
            for state, action, reward, next_state in trajectory:
                delta_sa = (
                    reward + self.discount * qv[next_state].max() - qv[state, action]
                )
                qv[state, action] += self.alpha() * delta_sa
        return qv

    def run(self, custom_agg: bool = False) -> None:
        self.infos = {"error_to_optimal": [], "number_of_regions": []}
        self.model.reward_matrix += 1

        counter = np.zeros((self.model.state_dim, self.model.action_dim))
        delta_average = np.zeros((self.model.state_dim, self.model.action_dim))

        partition = (
            QValuePartition(
                self.model,
                self.discount,
                SPARSE,
            )
            if custom_agg == False
            else self.get_initial_agg()
        )
        partition._compute_phi()
        contracted_q_value = np.zeros(
            (partition._number_of_regions(), self.model.action_dim)
        )
        # print("Number of regions:", partition._number_of_regions())
        iteration = 0

        while iteration < traj_count:
            if np.random.rand() < 0.05:
                break
            iteration += 1
            trajectory = generate_trajectory(
                self.model,
                traj_length,
                best_action,
                partition._partial_phi().dot(contracted_q_value),
            )
            trajectory_partition = convert_trajectory_to_region(trajectory, partition)
            for _ in range(replay_count):
                for ind in range(traj_length):
                    state, action, reward, next_state = trajectory[ind]
                    state_region, _, _, next_state_region = trajectory_partition[ind]
                    delta_sa = (
                        reward
                        + self.discount * contracted_q_value[next_state_region].max()
                        - contracted_q_value[state_region, action]
                    )
                    contracted_q_value[state_region, action] += (
                        self.alpha()
                        * delta_sa
                        / len(partition.states_in_region[state_region])
                    )
                    counter[state, action] += 1
                    n = counter[state, action]
                    delta_average[state, action] = (
                        n / (1 + n) * delta_average[state, action]
                        + 1 / (1 + n) * delta_sa
                    )

            if counter.sum() >= sample_before_division:
                partition._compute_weights_phi()
                qv = partition._partial_phi().dot(contracted_q_value)
                tqv = q_optimal_bellman_operator(self.model, qv, self.discount)

                contracted_q_value = (
                    partition.divide_all_regions_along_value_update_contracted_q_value(
                        tqv.max(axis=1),
                        epsilon_division,
                        contracted_q_value,
                    )
                )
                partition._compute_phi()
                delta_average[:, :] = 0
                counter[:, :] = 0
                # print("Dividing, number of regions:", partition._number_of_regions())

            self.add_error(partition._partial_phi().dot(contracted_q_value))
            # if iteration % (traj_count // 50) == 0:
            #     self.plot_q_value(partition._partial_phi().dot(contracted_q_value))

            if partition._number_of_regions() > 0.99 * self.model.state_dim:
                # print("Switching to Q-learning")
                break

        q_value = partition._partial_phi().dot(contracted_q_value)

        # print("switching to Q-learning")

        while iteration < traj_count:
            iteration += 1
            trajectory = generate_trajectory(
                self.model,
                traj_length,
                best_action,
                partition._partial_phi().dot(contracted_q_value),
            )
            for _ in range(replay_count):
                for state, action, reward, next_state in trajectory:
                    delta_sa = (
                        reward
                        + self.discount * q_value[next_state].max()
                        - q_value[state, action]
                    )
                    q_value[state, action] += self.alpha() * delta_sa

            self.add_error(q_value)

            exact = get_exact_value(self.model, self.discount)
            if iteration % 100000 == 0:
                plt.plot(q_value.max(axis=1), label='q')
                plt.plot(exact, label='V*')
                plt.legend()
                plt.show()

        # print(partition.states_in_region)
        # plt.plot(self.infos["number_of_regions"])
        # plt.title("Number of regions")
        # plt.show()

    def add_error(self, q_value):
        error = distance_to_optimal(q_value, self.model, self.discount)
        self.infos["error_to_optimal"].append(error)

    def plot_q_value(self, q_value: np.ndarray):
        """Plot the Q-value function."""
        # policy = q_value.argmax(axis=1)
        # print(is_policy_optimal(self.model, policy, self.discount))

        q_value = q_value.max(axis=1)
        optimal_value = get_exact_value(self.model, self.discount)
        sorted_indices = np.argsort(optimal_value)
        q_value = q_value[sorted_indices]
        optimal_value = optimal_value[sorted_indices]
        error = distance_to_optimal(q_value, self.model, self.discount)

        plt.figure(figsize=(10, 5))
        plt.title("Q-value - Error: {error:.4f}".format(error=error))
        plt.plot(q_value, label="Q-value")
        plt.plot(optimal_value, label="Optimal value")
        plt.xlabel("State")
        plt.ylabel("Value")
        plt.legend()
        plt.draw()
        plt.pause(1)
        plt.clf()

    def store_average_reward(self, trajectory: list) -> float:
        rewards = [reward for _, _, reward, _ in trajectory]
        avg_reward = np.mean(rewards)
        if "average_reward" not in self.infos:
            self.infos["average_reward"] = []
        self.infos["average_reward"].append(avg_reward)
        return np.mean(rewards)

    def get_initial_agg(self) -> GenericPartition:
        """Get the initial aggregation for the model."""
        _, solver = solve(
            model_name=self.model.name.split("_")[-1],
            solver_name="aggregated_qvi",
            state_dim=self.model.state_dim,
            action_dim=self.model.action_dim,
            discount=self.discount,
            precision=0.1,
        )
        return solver.partition
