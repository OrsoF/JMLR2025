import numpy as np
from utils.generic_model import GenericModel
from scipy.sparse import dok_array


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        self.action_dim = 4
        self.v_dim = int(np.sqrt(state_dim / 3))
        self.h_dim = 3 * self.v_dim
        self.state_dim = self.h_dim * self.v_dim

        self.name = "{}_{}_cliff".format(
            self.state_dim,
            self.action_dim,
        )

    def is_in_hole(self, state_index) -> bool:
        v = state_index // self.h_dim
        h = state_index % self.h_dim
        return v == 0 and h > 0 and h < self.h_dim - 1

    def get_next_state(self, state_index: int, action: int) -> int:
        v = state_index // self.h_dim
        h = state_index % self.h_dim

        if action == 0:
            next_v = min(v + 1, self.v_dim - 1)
            next_h = h

        elif action == 1:
            next_v = max(v - 1, 0)
            next_h = h

        elif action == 2:
            next_v = v
            next_h = min(h + 1, self.h_dim - 1)

        else:
            next_v = v
            next_h = max(h - 1, 0)

        return next_v * self.h_dim + next_h

    def _build_model(self):
        self.reward_matrix = np.zeros(
            (self.state_dim, self.action_dim), dtype=np.float16
        )
        self.transition_matrix = [
            dok_array((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        goal_state = self.h_dim - 1
        for state_index in range(self.state_dim):
            # Check if the state is a hole
            if self.is_in_hole(state_index):
                self.reward_matrix[state_index, :] = -1.0
                for action in range(self.action_dim):
                    self.transition_matrix[action][state_index, 0] = 1.0
            elif state_index == goal_state:
                self.reward_matrix[state_index, :] = 1.0
                for action in range(self.action_dim):
                    self.transition_matrix[action][state_index, goal_state] = 1.0
            else:
                for action in range(self.action_dim):
                    next_state = self.get_next_state(state_index, action)
                    self.transition_matrix[action][state_index, next_state] = 1.0

        # Convert transition matrices to sparse format
        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
