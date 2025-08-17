# From “Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition” by Tom Dietterich

import numpy as np
from scipy.sparse import dok_array
from utils.generic_model import GenericModel


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.grid_size = max(int(np.sqrt(state_dim // 20)), 1)
        self.state_dim = self.grid_size**2 * 20
        self.action_dim = 6

        self.name = "{}_{}_taxi".format(self.state_dim, self.action_dim)

    def build_state_space(self):
        self.state_list = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for passenger_position in range(5):
                    for destination in range(4):
                        self.state_list.append((x, y, passenger_position, destination))
        self.state_space = {state: ss for (ss, state) in enumerate(self.state_list)}

    def _build_model(self):
        self.build_state_space()

        self.transition_matrix: list = [
            dok_array((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        # The flags
        flags_on_map = [
            (0, 0),
            (self.grid_size - 1, 0),
            (0, self.grid_size - 1),
            (self.grid_size - 1, self.grid_size - 1),
        ]

        for ss1 in range(self.state_dim):
            for aa in range(self.action_dim):
                x, y, passenger_position, destination = self.state_list[ss1]

                if passenger_position == destination:
                    self.transition_matrix[aa][ss1, ss1] = 1.0
                    continue

                if aa == 4:  # Pickup passenger
                    if (
                        passenger_position <= 3
                        and (x, y) == flags_on_map[passenger_position]
                    ):  # Taxi is on passenger and can take it
                        passenger_position = 4 # Passenger is now in taxi
                        ss2 = self.state_space[(x, y, passenger_position, destination)]
                        self.transition_matrix[aa][ss1, ss2] = 1.0
                    else: # Taxi is not on passenger or passenger is already in taxi, no pickup
                        self.transition_matrix[aa][ss1, ss1] = 1.0

                    continue

                elif aa == 5:  # Delivering passenger
                    if (x, y) == flags_on_map[
                        destination
                    ] and passenger_position == 4:  # Passenger is in taxi, we are at destination
                        self.reward_matrix[ss1, aa] = 1.0

                        # Building the next state
                        ss2 = self.state_space[(x, y, destination, destination)]
                        self.transition_matrix[aa][ss1, ss2] = 1.0

                    else: # Delivering passenger failed
                        self.transition_matrix[aa][ss1, ss1] = 1.0

                    continue

                if aa == 0:  # North
                    y = max(0, y - 1)
                    ss2 = self.state_space[(x, y, passenger_position, destination)]
                elif aa == 1:
                    y = min(self.grid_size - 1, y + 1)
                    ss2 = self.state_space[(x, y, passenger_position, destination)]
                elif aa == 2:  # North
                    x = max(0, x - 1)
                    ss2 = self.state_space[(x, y, passenger_position, destination)]
                elif aa == 3:
                    x = min(self.grid_size - 1, x + 1)
                    ss2 = self.state_space[(x, y, passenger_position, destination)]

                self.transition_matrix[aa][ss1, ss2] = 1.0

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
