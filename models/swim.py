# @article{strehl2008analysis,
#   title={An analysis of model-based interval estimation for Markov decision processes},
#   author={Strehl, Alexander L and Littman, Michael L},
#   journal={Journal of Computer and System Sciences},
#   volume={74},
#   number={8},
#   pages={1309--1331},
#   year={2008},
#   publisher={Elsevier}
# }


import numpy as np
from utils.generic_model import GenericModel, SPARSE, NUMPY
from scipy.sparse import dok_matrix


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = max(3, state_dim)
        self.action_dim = 2

        self.name = "{}_{}_sim".format(self.state_dim, self.action_dim)

    def _build_model(self):
        self.transition_matrix: list = [
            dok_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros(
            (self.state_dim, self.action_dim), dtype=np.float16
        )

        for ss1 in range(self.state_dim):
            aa = 0
            if ss1 == 0:
                self.transition_matrix[aa][ss1, ss1] = 1.0

            else:
                self.transition_matrix[aa][ss1, ss1 - 1] = 1.0

            aa = 1
            if ss1 == 0:
                self.transition_matrix[aa][ss1, ss1] = 0.4
                self.transition_matrix[aa][ss1, ss1 + 1] = 0.6
                self.reward_matrix[ss1, aa] = 1e-3
            elif ss1 == self.state_dim - 1:
                self.transition_matrix[aa][ss1, ss1 - 1] = 0.4
                self.transition_matrix[aa][ss1, ss1] = 0.6
            else:
                self.transition_matrix[aa][ss1, ss1] = 0.6
                self.transition_matrix[aa][ss1, ss1 + 1] = 0.35
                self.transition_matrix[aa][ss1, ss1 - 1] = 0.05
                self.reward_matrix[ss1, aa] = 1

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
