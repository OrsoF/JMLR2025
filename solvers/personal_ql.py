import re
import numpy as np
from utils.generic_model import GenericModel, NUMPY, SPARSE
import time
from utils.partition_q_value import QValuePartition
from utils.calculus import generate_sars, generate_trajectory, optimal_bellman_residual
import matplotlib.pyplot as plt
import os
import tqdm
from utils.exact_value_function import distance_to_optimal
from params import traj_count, traj_length, learning_rate, decay_rate, best_action, replay_count

rng = np.random.default_rng(0)


class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
    ):
        # Class arguments
        self.model = model
        self.discount = discount
        self.model._convert_model(NUMPY)
        self.model._normalize_reward_matrix()
        self.name = "QL"
        self.steps_done_learning = 0

    def alpha(self) -> float:
        self.steps_done_learning += 1
        return 0.25* learning_rate / self.steps_done_learning**decay_rate

    def run(self):
        self.infos = {"error_to_optimal": []}
        q_value = np.zeros((self.model.state_dim, self.model.action_dim))

        for _ in range(traj_count):
            trajectory = generate_trajectory(
                self.model,
                traj_length,
                best_action,
                q_value,
            )
            for _ in range(replay_count):
                for elem in trajectory:
                    current_state, action, reward, next_state = elem
                    delta_sa = (
                        reward
                        + self.discount * q_value[next_state].max()
                        - q_value[current_state, action]
                    )
                    q_value[current_state, action] += self.alpha() * delta_sa
            error = distance_to_optimal(
                q_value,
                self.model,
                self.discount,
            )
            self.infos["error_to_optimal"].append(error)
