"""
Solve a Discounted MDP with Linear Programming (primal problem) using Gurobi Linear Solver.
"""

import numpy as np
import time
from gurobipy import GRB

from utils.data_management import gurobi_model_creation
from utils.generic_model import GenericModel
from utils.generic_solver import GenericSolver


class Solver(GenericSolver):
    def __init__(
        self,
        env: GenericModel,
        discount: float,
        final_precision: float,
    ):
        self.env = env
        self.name = "LPdual"

    def _create_variables(self):
        self.var = {}
        for ss in range(self.env.state_dim):
            self.var[ss] = self.model.addVar(
                vtype=GRB.CONTINUOUS, name="v({})".format(ss), lb=-10000, ub=10000
            )

    def _define_objective(self):
        self.model.setObjective(
            sum(self.var[ss] for ss in range(self.env.state_dim)), GRB.MINIMIZE
        )
        self.model.update()

    def _set_constraints(self):
        for ss1 in range(self.env.state_dim):
            for aa in range(self.env.action_dim):
                neighors_value_sum = sum(
                    self.env.transition_matrix[aa][ss1, ss2] * self.var[ss2]
                    for ss2 in range(self.env.state_dim)
                )
                self.model.addConstr(
                    self.var[ss1] - self.env.reward_matrix[ss1, aa] + neighors_value_sum
                    <= 0
                )
                self.model.addConstr(
                    self.var[ss1] - self.env.reward_matrix[ss1, aa] + neighors_value_sum
                    >= -1.0
                )

        # self.model.addConstr(0.0 <= self.var[0])
        # self.model.addConstr(self.var[0] <= 5000.0)
        self.model.update()

    def run(self):
        start_time = time.time()
        self.model = gurobi_model_creation()

        # Variables
        self._create_variables()

        # Objective
        self._define_objective()

        # Constraints
        self._set_constraints()

        self.model.optimize()

        self.runtime = time.time() - start_time

        self.value = np.array(self.model.x)

        print(self.value)
