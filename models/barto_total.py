# Can be found in Barto - Learning to act using real-time dynamic programming

import numpy as np
from utils.generic_model import GenericModel
from scipy.sparse import dok_matrix


def build_track(state_dim: int, max_speed: int) -> np.ndarray:
    max_speed = 2 * max_speed + 1
    size = int(np.sqrt(state_dim / max_speed / max_speed))
    track = np.ones((size, size))
    track[0, :] = 3
    track[:, 0] = 2
    track[0, 0] = 0
    return track


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        self._proba_fail_action: float = 0.1
        self._max_speed = 3

        state_dim = max(300, state_dim)
        self._track = build_track(state_dim, self._max_speed)

        self._X, self._Y = self._track.shape

        self._build_state_space()
        self._build_action_space()

        self.name = "{}_{}_barto_{}_{}".format(
            self.state_dim,
            self.action_dim,
            self._max_speed,
            self._proba_fail_action,
        )

    def _update_transition_matrix(
        self, aa: int, ax: int, ay: int, ss1: int, x: int, y: int, sx: int, sy: int
    ):
        if self._track[x, y] == 3:  # Finish line
            self.transition_matrix[aa][ss1, ss1] = 1.0
        else:
            # Not a finish line
            new_sx, new_sy = self.compute_new_speed(sx, sy, ax, ay)
            new_x, new_y, evolved = self.compute_new_position(x, y, sx, sy)

            if self._track[new_x, new_y] == 3:  # Finish line
                self.reward_matrix[ss1, aa] = 1.0

            # Avec probabilité p, la vitesse n'est pas modifiée
            s2_non_modified = self.state_space_index[(new_x, new_y, sx, sy)]
            self.transition_matrix[aa][ss1, s2_non_modified] += self._proba_fail_action
            # Avec probabilité 1-p, la vitesse est modifiée selon l'action
            s2_modified = self.state_space_index[(new_x, new_y, new_sx, new_sy)]
            self.transition_matrix[aa][ss1, s2_modified] += 1 - self._proba_fail_action

    def _build_model(self):
        self._init_matrices()

        for a, (ax, ay) in enumerate(self.action_space):
            for s1, (x, y, sx, sy) in enumerate(self.state_space):
                self._update_transition_matrix(a, ax, ay, s1, x, y, sx, sy)

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

    def _build_state_space(self):
        # A state is made of coordinate and speed : (x, y, sx, sy)
        self.state_space = [
            (x, y, sx, sy)
            for x in range(self._X)
            for y in range(self._Y)
            for sx in range(-self._max_speed, self._max_speed + 1)
            for sy in range(-self._max_speed, self._max_speed + 1)
            if self.check_on_track(x, y)
        ]

        self.state_space_index = {
            state: state_index for state_index, state in enumerate(self.state_space)
        }

        # state_dim is an integer
        self.state_dim = len(self.state_space)

    def _build_action_space(self):
        # An action is made of two increments (ax, ay) between -1 and +1
        self.action_space = [(ax, ay) for ax in range(-1, 2) for ay in range(-1, 2)]

        self.action_space_index = {
            action: action_index
            for action_index, action in enumerate(self.action_space)
        }

        self.action_dim = len(self.action_space)

    def _init_matrices(self):
        self._build_state_space()
        self._build_action_space()
        self.transition_matrix: list = [
            dok_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros(
            (self.state_dim, self.action_dim), dtype=np.float16
        )

        # Liste des états de départ
        self.start_states = [
            state
            for state in range(self.state_dim)
            if (
                abs(self._track[self.state_space[state][:2]] - 2.0)
                <= 0.01  # Etat de départ
                and self.state_space[state][2:] == (0, 0)  # La vitesse est nulle
            )
        ]
        # Proba d'être dans un état de départ
        self.proba_start_states = 1 / len(self.start_states)

    def check_on_track(self, x, y):
        """
        Checks if the point of coordinates (x, y) is on the track
        """
        return (0 <= x < self._X and 0 <= y < self._Y) and bool(self._track[x, y])

    def compute_new_speed(self, sx, sy, ax, ay):
        """
        From the speed (sx, sy), compute the next
        speed using the action (ax, ay)
        """
        new_sx = max(0, min(self._max_speed, sx + ax))
        new_sy = max(0, min(self._max_speed, sy + ay))
        return new_sx, new_sy

    def evolve_x(self, x, y, action):
        assert self.check_on_track(x, y)
        if self.check_on_track(x + action, y):
            return x + action, True
        else:
            return x, False

    def evolve_y(self, x, y, action):
        assert self.check_on_track(x, y)
        if self.check_on_track(x, y + action):
            return y + action, True
        else:
            return y, False

    def compute_new_position(self, x, y, sx, sy):
        assert self.check_on_track(x, y)
        speed_x, speed_y = sx, sy
        evolved_x, evolved_y = True, True
        for _ in range(abs(sx) + abs(sy)):
            if speed_x > 0:
                x, evolved_x = self.evolve_x(x, y, 1)
                speed_x -= 1

            elif speed_x < 0:
                x, evolved_x = self.evolve_x(x, y, -1)
                speed_x += 1

            if speed_y > 0:
                y, evolved_y = self.evolve_y(x, y, 1)
                speed_y -= 1

            if speed_y != 0:
                y, evolved_y = self.evolve_y(x, y, -1)
                speed_y += 1

        return x, y, evolved_x and evolved_y
