# Just like QL, but with memory replay.
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import trange, tqdm
import time

from utils.partition_q_value import QValuePartition
from utils.calculus import generate_sars, generate_trajectory, optimal_bellman_residual
from utils.generic_model import GenericModel, NUMPY, SPARSE
from utils.exact_value_function import distance_to_optimal

rng = np.random.default_rng(0)


class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float = 1e-2,
        traj_count: int = 500,
        traj_length: int = 200,
        leaning_rate: float = 0.5,
        decay_rate: float = 0.5,
        best_action: float = 0.95,
        get_infos: bool = True,
    ):
        # Class arguments
        self.model = model
        self.discount = discount
        self.model._convert_model(NUMPY)
        self.name = "QL"
        self.steps_done_learning = 0

        self.learning_rate = leaning_rate
        self.decay_rate = decay_rate

        self.traj_count = traj_count
        self.traj_length = traj_length

        self.best_action = best_action

        self.get_infos = get_infos

    def alpha(self) -> float:
        self.steps_done_learning += 1
        return self.learning_rate * 10 / (100 + self.steps_done_learning) ** self.decay_rate

    def run(self):
        start_time = time.time()
        if self.get_infos:
            self.infos = {"error_to_optimal": []}

        q_value = np.zeros((self.model.state_dim, self.model.action_dim))

        for _ in trange(self.traj_count):
            trajectory = generate_trajectory(
                self.model, self.traj_length, self.best_action, q_value
            )
            for _ in range(50): # Number of replays
                for elem in trajectory:
                    current_state, action, reward, next_state = elem
                    q_value[current_state, action] += self.alpha() * (
                        reward
                        + self.discount * q_value[next_state].max()
                        - q_value[current_state, action]
                    )

            if self.get_infos:
                error = optimal_bellman_residual(self.model, q_value.max(axis=1), self.discount)
                self.infos["error_to_optimal"].append(error)


        self.runtime = time.time() - start_time
        self.value = q_value.max(axis=1)
