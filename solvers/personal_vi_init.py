import numpy as np
from utils.generic_model import GenericModel, NUMPY, SPARSE
import time


class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
        mode: str = SPARSE,
    ):
        self.model = model
        self.discount = discount
        self.epsilon = final_precision
        self.mode = mode

        self.model._convert_model(self.mode)
        self.name = "VI"

        # print("Change the product in at in Personal Value Iteration")

    def bellman_operator(self, value):
        q_value = np.zeros((self.model.state_dim, self.model.action_dim))

        for aa in range(self.model.action_dim):
            q_value[:, aa] = self.model.reward_matrix[
                :, aa
            ] + self.discount * self.model.transition_matrix[aa].dot(value)

        return q_value.max(axis=1)

    def run(self):
        start_time = time.time()
        self.value = np.zeros((self.model.state_dim))
        # self.value = self.model.reward_matrix.max(axis=1) / (1 - self.discount)

        while True:
            new_value = self.bellman_operator(self.value)
            bellman_residual = np.linalg.norm(new_value - self.value, ord=np.inf)
            if bellman_residual < self.epsilon * (1 - self.discount):
                self.value = new_value
                break
            self.value = new_value

        self.runtime = time.time() - start_time
