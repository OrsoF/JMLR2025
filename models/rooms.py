# The model can be found described in Hengst - Hierarchical approaches

import numpy as np
from utils.generic_model import GenericModel
from scipy.sparse import dok_array

param_list = []


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        state_dim = max(100, state_dim)
        self.size_of_one_room = int(np.sqrt(state_dim) / 2.0)
        self.n_doors = 1

        self.action_dim = 4
        self.state_dim = 4 * self.size_of_one_room**2

        self.name = "{}_{}_rooms".format(self.state_dim, self.action_dim)

    def _build_model(self):
        self.side_size = self.size_of_one_room * 2

        self.door_step = self.size_of_one_room / (self.n_doors + 1)

        self.doors_indices = [
            int(i * self.door_step) for i in range(1, self.n_doors + 1)
        ] + [
            self.size_of_one_room + int(i * self.door_step)
            for i in range(1, self.n_doors + 1)
        ]

        self.reward_matrix = -np.ones((self.state_dim, self.action_dim))
        self.transition_matrix: list = [
            dok_array((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.transition_matrix = self._add_displacements(self.transition_matrix)
        self.transition_matrix = self._add_walls(self.transition_matrix)
        self.transition_matrix, self.reward_matrix = self._add_exit(
            self.transition_matrix, self.reward_matrix
        )

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

    def _coord(self, ss: int):
        assert ss < self.side_size**2
        return ss // self.side_size, ss % self.side_size

    def _state(self, i: int, j: int):
        return i * self.side_size + j

    def _next_state(self, i, j, direction):
        assert isinstance(direction, int) and 0 <= direction < 4
        if direction == 0:  # North
            return i - 1, j
        elif direction == 1:  # South
            return i + 1, j
        elif direction == 2:  # East
            return i, j + 1
        else:  # West
            return i, j - 1

    def _is_in_grid(self, i: int, j: int):
        return 0 <= i < self.side_size and 0 <= j < self.side_size

    def _add_displacements(self, transition_matrix: list):
        for a in range(self.action_dim):
            for ss in range(self.state_dim):
                i, j = self._coord(ss)
                next_coord = self._next_state(i, j, a)
                if self._is_in_grid(*next_coord):
                    transition_matrix[a][ss, ss] = 0.2
                    transition_matrix[a][ss, self._state(*next_coord)] = 0.8
                else:
                    transition_matrix[a][ss, ss] = 1.0
        return transition_matrix

    def _add_wall_element(
        self, ss1: int, ss2: int, transition_matrix: list, direction: str
    ):
        assert direction in ["horizontal", "vertical"]
        i, j = self._coord(ss1)
        k, l = self._coord(ss2)
        assert (i - k) ** 2 + (j - l) ** 2 == 1
        if direction == "vertical":
            assert j < l
            transition_matrix[2][ss1, ss1] = 1.0
            transition_matrix[2][ss1, ss2] = 0.0
            transition_matrix[3][ss2, ss2] = 1.0
            transition_matrix[3][ss2, ss1] = 0.0
        else:
            assert i < k
            transition_matrix[1][ss1, ss1] = 1.0
            transition_matrix[1][ss1, ss2] = 0.0
            transition_matrix[0][ss2, ss2] = 1.0
            transition_matrix[0][ss2, ss1] = 0.0
        return transition_matrix

    def _add_walls(self, transition_matrix: list):
        # Horizontal wall
        direction = "horizontal"
        i, k = self.side_size // 2 - 1, self.side_size // 2
        for index in range(self.side_size):
            if index not in self.doors_indices:
                j, l = index, index
                ss1, ss2 = self._state(i, j), self._state(k, l)
                transition_matrix = self._add_wall_element(
                    ss1, ss2, transition_matrix, direction
                )

        # Vertical wall
        direction = "vertical"
        j, l = self.side_size // 2 - 1, self.side_size // 2
        for index in range(self.side_size):
            if index not in self.doors_indices:
                i, k = index, index
                ss1, ss2 = self._state(i, j), self._state(k, l)
                transition_matrix = self._add_wall_element(
                    ss1, ss2, transition_matrix, direction
                )

        return transition_matrix

    def _add_exit(self, transition_matrix: list, reward_matrix: np.ndarray):
        """Add exit at the third top left box."""
        i, j = 0, self.side_size // 4
        exit_state = self._state(i, j)
        for a in range(self.action_dim):
            transition_matrix[a][exit_state, :] = 0
            transition_matrix[a][exit_state, exit_state] = 1
        reward_matrix[exit_state, :] = 0

        return transition_matrix, reward_matrix
