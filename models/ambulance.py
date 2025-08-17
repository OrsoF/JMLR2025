# Model selected from "Collaborative learning in networks", Mason, W. and Watts

import numpy as np
from utils.generic_model import GenericModel, SPARSE, NUMPY
from scipy.sparse import dok_matrix
from scipy.stats import beta
from tqdm import trange, tqdm


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.name = "{}_{}_ambulance".format(self.state_dim, self.action_dim)

    def _reward_function(self, ss, aa):
        ss /= self.state_dim - 1
        aa /= self.action_dim - 1
        return 1 - abs(ss - aa)

    def _transition_function(self, ss2: int):
        ss2 /= self.state_dim - 1
        return beta.pdf(ss2, 5, 2)

    def _build_model(self):
        self.transition_matrix = [
            dok_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        normalization_constant = 1 / sum(
            self._transition_function(ss2) for ss2 in range(self.state_dim)
        )

        for ss2 in trange(self.state_dim):
            transition = normalization_constant * self._transition_function(ss2)
            for aa in range(self.action_dim):
                self.transition_matrix[aa][:, ss2] = transition

        for ss1 in trange(self.state_dim):
            prop_s = ss1 / (self.state_dim - 1)
            for aa in range(self.action_dim):
                prop_a = aa / (self.action_dim - 1)
                self.reward_matrix[ss1, aa] = 1 - abs(prop_s - prop_a)

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
