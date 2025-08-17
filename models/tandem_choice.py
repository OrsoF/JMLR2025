# The model can be found in Tournaire - Factored reinforcement learning for auto-scaling in tandem queues

import numpy as np
from utils.generic_model import GenericModel
from scipy.sparse import dok_array
import itertools


def space_dictionary(list_of_tuples: list):
    ranges = (range(*dimension_tuple) for dimension_tuple in list_of_tuples)
    list_of_ranges = tuple(ranges)
    return {elem: i for i, elem in enumerate(itertools.product(*list_of_ranges))}


def params_match_state_dim(state_dim: int, max_index=50) -> tuple:
    """Get the tandem queue parameters that match the best state_dim."""
    distance = np.inf
    b1, b2, k1, k2 = 0, 0, 0, 0
    for b1_ in range(1, max_index):
        b2_ = b1_
        for k1_ in range(1, max_index):
            k2_ = k1_
            dim = np.prod([b1_ + 1, b2_ + 1, k1_, k2_])
            if abs(dim - state_dim) < distance:
                distance = abs(dim - state_dim)
                b1, b2, k1, k2 = b1_, b2_, k1_, k2_
    return b1, b2, k1, k2, int(np.prod([b1 + 1, b2 + 1, k1, k2]))


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        action_dim = max(action_dim, 9)
        self.actions_per_server = int(np.sqrt(action_dim))

        self.arrival_rate_lambda = 0.6
        self.mu_1, self.mu_2 = 0.2, 0.2
        self.B1, self.B2, self.K1, self.K2, self.state_dim = params_match_state_dim(
            state_dim
        )

        self.CA = 1.0
        self.CD = 1.0
        self.CH = 1.0
        self.CS = 1.0
        self.CR = 1.0
        self.discount = 0.99

        print()
        print("Current discount in tandem model: {}".format(self.discount))
        print()

        self.lambda_tilde = (
            self.arrival_rate_lambda + self.K1 * self.mu_1 + self.K2 * self.mu_2
        )

        self.action_dim = int(self.K1 * self.K2)
        self.name = "{}_{}_tandem_choice".format(self.state_dim, self.action_dim)

    def _build_model(self):
        # State space
        self.state_space_tuples = [
            (0, self.B1 + 1),
            (0, self.B2 + 1),
            (1, self.K1 + 1),
            (1, self.K2 + 1),
        ]
        self.state_encoding = space_dictionary(self.state_space_tuples)
        self.state_list = list(self.state_encoding.keys())

        # Action space
        self.action_space_tuples = [
            (1, self.K1 + 1),
            (1, self.K2 + 1),
        ]
        self.action_encoding = space_dictionary(self.action_space_tuples)
        self.action_list = list(self.action_encoding.keys())

        # Transition and reward matrices
        self.transition_matrix: list = [
            dok_array((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.empty((self.state_dim, self.action_dim))
        self.build_transition_reward_matrices()

    def n1(self, k):
        return min(max(1, k), self.K1)

    def n2(self, k):
        return min(max(1, k), self.K2)

    def lambda_function(self, s, a):
        m1, m2, k1, k2 = self.state_list[s]
        a1, a2 = self.action_decoding[a]
        return (
            self.arrival_rate_lambda
            + self.mu_1 * min(m1, self.n1(k1 + a1))
            + self.mu_2 * min(m2, self.n2(k2 + a2))
        )

    def c1(self, s, a):
        m1, _, k1, _ = self.state_list[s]
        a1, _ = self.action_decoding[a]
        return self.n1(k1 + a1) * self.CS + m1 * self.CH

    def c2(self, s, a):
        _, m2, _, k2 = self.state_list[s]
        _, a2 = self.action_decoding[a]
        return self.n2(k2 + a2) * self.CS + m2 * self.CH

    def h1(self, s, a):
        m1, _, _, _ = self.state_list[s]
        a1, _ = self.action_decoding[a]
        return (
            self.CA * int(a1 == 1)
            + self.CD * int(a1 == -1)
            + self.arrival_rate_lambda
            * self.CR
            * int(m1 == self.B1 - 1)
            / (self.lambda_function(s, a) + self.discount)
        )

    def h2(self, s, a):
        m1, m2, k1, _ = self.state_list[s]
        a1, a2 = self.action_decoding[a]
        return (
            self.CA * int(a2 == 1)
            + self.CD * int(a2 == -1)
            + min(m1, self.n1(k1 + a1))
            * self.mu_1
            * self.CR
            * int(m2 == self.B2 - 1)
            / (self.lambda_function(s, a) + self.discount)
        )

    def s1p(self, s, a):
        m1, m2, k1, k2 = self.state_list[s]
        a1, a2 = self.action_decoding[a]
        return self.state_encoding[
            (min(m1 + 1, self.B1), m2, self.n1(k1 + a1), self.n2(k2 + a2))
        ]

    def s2p(self, s, a):
        m1, m2, k1, k2 = self.state_list[s]
        a1, a2 = self.action_decoding[a]
        return self.state_encoding[
            (max(m1 - 1, 0), min(m2 + 1, self.B2), self.n1(k1 + a1), self.n2(k2 + a2))
        ]

    def s3p(self, s, a):
        m1, m2, k1, k2 = self.state_list[s]
        a1, a2 = self.action_decoding[a]
        return self.state_encoding[
            (m1, max(m2 - 1, 0), self.n1(k1 + a1), self.n2(k2 + a2))
        ]

    def build_transition_reward_matrices(self):
        for aa in range(self.action_dim):
            for ss1 in range(self.state_dim):
                state = self.state_list[ss1]
                action = self.action_list[aa]

                m1, m2, k1, k2 = state
                a1, a2 = action  # a1 : number of VMs in queue 1

                # First case : arrival in queue 1 with rate lambda
                next_state_1 = (min(m1 + 1, self.B1), m2, a1, a2)
                ss2_1 = self.state_encoding[next_state_1]
                rate_1 = self.arrival_rate_lambda

                # Second, departure from queue 1 and entry in queue 2
                next_state_2 = (max(m1 - 1, 0), min(m2 + 1, self.B2), a1, a2)
                ss2_2 = self.state_encoding[next_state_2]
                rate_2 = self.mu_1 * min(m1, a1)

                # Third, departure from queue 2
                next_state_3 = (m1, max(m2 - 1, 0), a1, a2)
                ss2_3 = self.state_encoding[next_state_3]
                rate_3 = self.mu_2 * min(m2, a2)

                lambda_cap = rate_1 + rate_2 + rate_3

                self.transition_matrix[aa][ss1, ss2_1] += rate_1 / self.lambda_tilde
                self.transition_matrix[aa][ss1, ss2_2] += rate_2 / self.lambda_tilde
                self.transition_matrix[aa][ss1, ss2_3] += rate_3 / self.lambda_tilde
                self.transition_matrix[aa][ss1, ss1] += (
                    1 - lambda_cap / self.lambda_tilde
                )

                reward = (
                    (self.CS * (a1 + a2) + self.CH * (m1 + m2))
                    / (lambda_cap + self.discount)
                    + self.CA * (max(0, a1 - k1) + max(0, a2 - k2))
                    + self.CD * (min(0, k1 - a1) + min(0, k2 - a2))
                    + self.CR
                    / (lambda_cap + self.discount)
                    * (
                        self.arrival_rate_lambda * float(m1 == self.B1)
                        + self.mu_1 * min(m1, a1) * float(m2 == self.B2)
                    )
                )
                reward_tilde = (
                    reward
                    * (lambda_cap + self.discount)
                    / (self.lambda_tilde + self.discount)
                )

                self.reward_matrix[ss1, aa] = reward_tilde

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
