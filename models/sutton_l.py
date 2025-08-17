# Model from "Reinforcement Learning, An Introduction", Sutton and Barto, Exercise 5.12 p.111 Edition 2018

import numpy as np
from utils.generic_model import GenericModel
from scipy.sparse import dok_array

# fmt: off

track_L = np.array([
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
])


track_R = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],     
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],     
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],     
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0], 
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0], 
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0], 
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0], 
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],     
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],    
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],    
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],    
    [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],    
    [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],   
    [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],     
    [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],     
    [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],     
    [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],     
    [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],     
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],    
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],    
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
])


# fmt: on


def build_track(state_dim: int) -> np.ndarray:
    size = int(np.sqrt(state_dim / 35))
    track = np.ones((size, size))
    track[-1, -1] = 0
    return track


class Model(GenericModel):
    def __init__(
        self, state_dim: int, action_dim: int, custom_track: np.ndarray = track_L
    ):
        state_dim = max(state_dim, 500)
        if custom_track is None:
            self.track = build_track(state_dim)
        else:
            self.track = custom_track

        self.state_dim = np.count_nonzero(self.track) * 36

        self.action_dim = 9
        self.name = "{}_{}_sutton".format(self.state_dim, self.action_dim)

    def _build_state_space(self):
        self.state_space = []
        for x in range(self.X):
            for y in range(self.Y):
                for sx in range(6):
                    for sy in range(6):
                        self.state_space = self._update_state_space_if_on_track(
                            self.state_space, sx, sy, x, y
                        )

    def _update_state_space_if_on_track(
        self, state_space: list, sx: int, sy: int, x: int, y: int
    ) -> list:
        if sx + sy and self._check_on_track(x, y):
            state_space.append((x, y, sx, sy))
        return state_space

    def _build_state_index(self):
        self.state_space_index = {
            state: state_index for state_index, state in enumerate(self.state_space)
        }

    def _build_matrices(self):
        self.Y, self.X = self.track.shape
        # A state is made of coordinate and speed : (x, y, sx, sy)
        self._build_state_space()
        self._build_state_index()

        # An action is made of two increments (ax, ay)
        self.action_space = {(ax, ay) for ax in range(-1, 2) for ay in range(-1, 2)}
        self.action_space_index = {
            action: action_index
            for action_index, action in enumerate(self.action_space)
        }
        self.state_dim = len(self.state_space)
        self.action_dim = len(self.action_space)

    def _build_model(self):
        self._build_matrices()

        self.transition_matrix: list = [
            dok_array((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = -1 * np.ones(
            (self.state_dim, self.action_dim), dtype=np.float16
        )
        for a, (ax, ay) in enumerate(self.action_space):
            for s1, (x, y, sx, sy) in enumerate(self.state_space):
                new_sx, new_sy = self._compute_new_speed(sx, sy, ax, ay)
                new_x, new_y, out_track = self._compute_new_position(x, y, sx, sy)
                if out_track:
                    self.reward_matrix[s1, a] = -5

                s2 = self.state_space_index[(new_x, new_y, new_sx, new_sy)]
                self.transition_matrix[a][s1, s2] += 0.5

                if new_x < self.X - 1 and self._check_on_track(new_x + 1, new_y):
                    s3 = self.state_space_index[(new_x + 1, new_y, new_sx, new_sy)]
                    self.transition_matrix[a][s1, s3] = 0.25
                else:
                    self.transition_matrix[a][s1, s2] += 0.25

                if new_y < self.Y - 1 and self._check_on_track(new_x, new_y + 1):
                    s4 = self.state_space_index[(new_x, new_y + 1, new_sx, new_sy)]
                    self.transition_matrix[a][s1, s4] = 0.25
                else:
                    self.transition_matrix[a][s1, s2] += 0.25

                if x == self.X:
                    self.reward_matrix[s1, a] = 0

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

    def _check_on_track(self, x, y):
        """
        Checks if the point of coordinates (x, y) is on the track
        """
        if not (x < self.X and y < self.Y):
            return False
        else:
            return bool(self.track[self.Y - y - 1, x])

    def _compute_new_speed(self, sx, sy, ax, ay):
        """
        From the speed (sx, sy), compute the next
        speed using the action (ax, ay)
        """
        new_sx = max(0, min(5, sx + ax))
        new_sy = max(0, min(5, sy + ay))
        if new_sx == 0 and new_sy == 0:
            return sx, sy
        else:
            return new_sx, new_sy

    def _evolve_x(self, x, y):
        assert self._check_on_track(x, y)
        if self._check_on_track(x + 1, y):
            return x + 1, True
        else:
            return x, False

    def _evolve_y(self, x, y):
        assert self._check_on_track(x, y)
        if self._check_on_track(x, y + 1):
            return y + 1, True
        else:
            return y, False

    def _compute_new_position(self, x, y, sx, sy):
        """
        For a position (x, y) and a speed (sx, sy),
        compute new position (x, y).
        """
        assert self._check_on_track(x, y)
        speed_x, speed_y = sx, sy
        evolved_x, evolved_y = True, True
        for _ in range(sx + sy):
            if speed_x > 0:
                x, evolved_x = self._evolve_x(x, y)
                speed_x -= 1

            if speed_y > 0:
                y, evolved_y = self._evolve_y(x, y)
                speed_y -= 1

        return x, y, evolved_x and evolved_y
