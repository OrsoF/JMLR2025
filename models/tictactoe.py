from socket import gaierror
import numpy as np
from scipy.sparse import dok_array
from utils.generic_model import GenericModel
from tqdm import trange

def revert_grid(grid: list) -> list:
    for index in range(len(grid)):
        if grid[index] == 1:
            grid[index] = 2
        elif grid[index] == 2:
            grid[index] = 1
    return grid


def is_constant_equal(arr: np.ndarray, value):
    return arr.max() == arr.min() == value


def is_grid_won_by_1(grid: list) -> bool:
    shape = int(np.sqrt(len(grid)))
    grid: np.ndarray = np.array(grid).reshape((shape, shape))
    diag, anti_diag = grid.diagonal(), np.fliplr(grid).diagonal()
    if is_constant_equal(diag, 1) or is_constant_equal(anti_diag, 1):
        return True
    for row in grid:
        if is_constant_equal(row, 1):
            return True
    for col in grid.T:
        if is_constant_equal(col, 1):
            return True
    return False


def is_grid_won_by_2(grid: list) -> bool:
    shape = int(np.sqrt(len(grid)))
    grid: np.ndarray = np.array(grid).reshape((shape, shape))
    diag, anti_diag = grid.diagonal(), np.fliplr(grid).diagonal()
    if is_constant_equal(diag, 2) or is_constant_equal(anti_diag, 2):
        return True
    for row in grid:
        if is_constant_equal(row, 2):
            return True
    for col in grid.T:
        if is_constant_equal(col, 2):
            return True
    return False


def is_grid_tie(grid: list):
    return np.min(grid) > 0


def evolve_grid(grid: list, action: int) -> list:
    if grid[action] != 0:
        return grid
    elif is_grid_won_by_1(grid) or is_grid_won_by_2(grid):
        return grid
    else:
        grid = grid.copy()
        grid[action] = 1
        return revert_grid(grid)


def is_game_finished(grid) -> bool:
    return is_grid_won_by_1(grid) or is_grid_won_by_2(grid) or is_grid_tie(grid)


def render_grid(grid: list):
    shape = int(np.sqrt(len(grid)))
    grid: np.ndarray = np.array(grid).reshape((shape, shape))
    print(grid)


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.action_dim = 9

        self.build_state_space()
        self.name = "{}_{}_tictactoe".format(self.state_dim, self.action_dim)

    def build_state_space(self):
        start_grid = [0 for _ in range(9)]

        self.state_list = [start_grid]

        while True:
            old_len = len(self.state_list)
            for state in self.state_list.copy():
                for action in range(self.action_dim):
                    next_state = evolve_grid(state, action)

                    if (state != next_state) and (next_state not in self.state_list):
                        self.state_list.append(next_state)

            if len(self.state_list) == old_len:
                break

        self.state_space = {
            tuple(state): ss for (ss, state) in enumerate(self.state_list)
        }
        self.state_dim = len(self.state_list)

    def _build_model(self):
        self.build_state_space()

        self.transition_matrix: list = [
            dok_array((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        for ss1 in trange(self.state_dim):
            state = self.state_list[ss1]
            for aa in range(self.action_dim):
                if is_grid_won_by_1(state):
                    self.transition_matrix[aa][ss1, ss1] = 1.0
                    self.reward_matrix[ss1, aa] = 1
                    continue
                elif is_grid_won_by_2(state):
                    self.transition_matrix[aa][ss1, ss1] = 1.0
                    self.reward_matrix[ss1, aa] = -1
                    continue
                elif is_grid_tie(state):
                    self.transition_matrix[aa][ss1, ss1] = 1.0
                    continue

                next_state = evolve_grid(state, aa)
                ss2 = self.state_space[tuple(next_state)]
                self.transition_matrix[aa][ss1, ss2] = 1.0

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
