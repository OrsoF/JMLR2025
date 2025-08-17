import numpy as np
from scipy.sparse import random
from utils.generic_model import GenericModel


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sparsity_transition = 0.85
        self.reward_std = 1.0

        self.check_parameters()

        self.name = "{}_{}_garnet_{}_{}".format(
            self.state_dim,
            self.action_dim,
            self.sparsity_transition,
            self.reward_std,
        )

    def check_parameters(self):
        assert isinstance(self.state_dim, int) and self.state_dim > 0
        assert isinstance(self.action_dim, int) and self.action_dim > 0
        assert 0.0 <= self.sparsity_transition <= 1.0
        assert self.reward_std > 0.0

    def _build_model(self):
        random_generator = np.random.default_rng(seed=0)
        self.reward_matrix = self.reward_std * random_generator.standard_normal(
            size=(self.state_dim, self.action_dim)
        )
        self.transition_matrix = [
            random(
                self.state_dim,
                self.state_dim,
                density=self.sparsity_transition,
                format="lil",
                data_rvs=np.random.rand,
            )
            for _ in range(self.action_dim)
        ]
        for aa in range(self.action_dim):
            for ss in range(self.state_dim):
                self.transition_matrix[aa][ss, 0] += 1e-6
                self.transition_matrix[aa][ss] *= (
                    1 / self.transition_matrix[aa][ss].sum()
                )

                self.transition_matrix[aa][ss, 0] += (
                    1 - self.transition_matrix[aa][ss].sum()
                )

        self.transition_matrix = [
            transition.tocsr() for transition in self.transition_matrix
        ]
