"""
This code allows to slice a state space progressively 
along an evolving value function.
"""

import numpy as np
from time import time

from utils.generic_model import GenericModel, NUMPY, SPARSE
from utils.partition_value import ValuePartition
from utils.calculus import (
    optimal_bellman_operator,
    norminf,
)


class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
        verbose: bool = False,
        iter_agg: int = 10,
        iter_bellman: int = 100,
        mode: str = SPARSE,
    ):
        # Class arguments
        self.model = model
        self.discount = discount
        self.epsilon = final_precision
        self.verbose = verbose
        self.iter_agg = iter_agg
        self.iter_bellman = iter_bellman
        self.mode = mode

        self.model._convert_model(self.mode)
        self.name = "Chen"

        # Variables
        self.partition = ValuePartition(self.model, self.discount, self.mode)
        self.contracted_value: np.ndarray

        self.alpha_function = lambda t: 0.001 / (t + 1)

    def run(self):
        start_time = time()
        tolerance = self.epsilon * (1 - self.discount) if self.discount < 1.0 else self.epsilon
        self.value = np.zeros((self.model.state_dim))

        n = 1
        while True:
            n += 1
            for _ in range(self.iter_bellman):
                new_value = optimal_bellman_operator(
                    self.model, self.value, self.discount
                )
                if norminf(new_value - self.value) < tolerance:
                    self.runtime = time() - start_time
                    return
                self.value = new_value

            # self.partition = ValuePartition(self.model, self.discount, self.mode)
            self.partition.__init__(self.model, self.discount, self.mode)
            self.partition.divide_all_regions_along_value_update_contracted_value(
                self.value,
                self.epsilon,
                np.zeros((len(self.partition.states_in_region))),
            )
            self.partition._compute_weights_phi()
            self.contracted_value = self.partition.weights.dot(self.value)
            alpha = self.alpha_function(n * (self.iter_agg + self.iter_bellman))

            for _ in range(self.iter_agg):
                self.extended_value = (
                    self.partition._partial_phi().dot(self.contracted_value)
                )
                for k in range(len(self.partition.states_in_region)):
                    ss = np.random.choice(self.partition.states_in_region[k])
                    # tsv_action = np.zeros((self.model.action_dim))
                    # for aa in range(self.model.action_dim):
                    #     tsv_action[aa] = self.model.reward_matrix[
                    #         ss, aa
                    #     ] + self.discount * self.model.transition_matrix[aa][
                    #         [ss], :
                    #     ].dot(
                    #         self.extended_value
                    #     )

                    tsv_action = [self.model.reward_matrix[
                            ss, aa
                        ] + self.discount * self.model.transition_matrix[aa][
                            [ss], :
                        ].dot(
                            self.extended_value
                        ) for aa in range(self.model.action_dim)]

                    self.contracted_value[k] = (1 - alpha) * self.contracted_value[
                        k
                    ] + alpha * max(tsv_action)

            self.value = self.partition._partial_phi().dot(self.contracted_value)
