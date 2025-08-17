# Model selected from "Collaborative learning in networks", Mason, W. and Watts

import numpy as np
from utils.generic_model import GenericModel, SPARSE, NUMPY
from scipy.sparse import dok_matrix


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = state_dim

        self._reward_type = "laplace"  # "quadratic"

        self.l = 1
        self.c = 0.75

        self.name = "{}_{}_oil".format(self.state_dim, self.action_dim)

    def _laplace_survey_function(self, ss):
        return np.exp(-self.l * abs(ss - self.c))

    def _quadratic_survey_function(self, ss):
        return 1 - self.l * (ss - self.c) ** 2

    def _reward_function(self, ss: int, aa: int) -> float:
        ss /= self.state_dim - 1
        aa /= self.action_dim - 1
        if self._reward_type == "laplace":
            fx = self._laplace_survey_function(ss)
        else:
            fx = self._quadratic_survey_function(ss)
        return max(0, fx - abs(ss - aa))

    def _build_model(self):
        self.transition_matrix: list = [
            dok_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros(
            (self.state_dim, self.action_dim), dtype=np.float16
        )

        for ss1 in range(self.state_dim):
            for aa in range(self.action_dim):
                self.transition_matrix[aa][ss1, aa] = 1.0

                self.reward_matrix[ss1, aa] = self._reward_function(ss1, aa)

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
