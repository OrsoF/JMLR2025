"""
Solve a Discounted MDP with Linear Programming (dual problem) using Gurobi Linear Solver.
"""

from gurobipy import LinExpr, GRB
import numpy as np
import time

from utils.generic_model import GenericModel
from utils.generic_solver import GenericSolver
from utils.calculus import policy_evaluation
from utils.data_management import gurobi_model_creation


class Solver(GenericSolver):
    def __init__(
        self,
        env: GenericModel,
        discount: float,
        final_precision: float,
    ):
        self.env = env
        self.name = "LPprimal"
        self.transition_matrix_is_sparse: bool = not isinstance(
            self.env.transition_matrix, np.ndarray
        )

    def _create_variables(self):
        # Dual variables
        self.var = {}
        for s in range(self.env.state_dim):
            for a in range(self.env.action_dim):
                self.var[(s, a)] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)
        self.model.update()

    def _define_objective(self):
        self.obj = LinExpr()
        for s in range(self.env.state_dim):
            for a in range(self.env.action_dim):
                self.obj += self.env.reward_matrix[s, a] * self.var[(s, a)]
        self.model.setObjective(self.obj, GRB.MAXIMIZE)

    def _set_constraints(self):
        for ss1 in range(self.env.state_dim):
            sum_value_over_action = sum(
                self.var[(ss1, aa)] for aa in range(self.env.action_dim)
            )
            sum_value_neighbors = sum(
                self.env.transition_matrix[aa][sp, ss1] * self.var[(sp, aa)]
                for aa in range(self.env.action_dim)
                for sp in range(self.env.state_dim)
            )
            self.model.addConstr(
                (sum_value_over_action - sum_value_neighbors - 1 <= 0),
                "Contrainte",
            )
            self.model.addConstr(
                (sum_value_over_action - sum_value_neighbors - 1 >= -1),
                "Contrainte",
            )

    def run(self):
        start_time = time.time()
        self.model = gurobi_model_creation()

        # Primal variables
        self._create_variables()

        # Objective definition
        self._define_objective()

        # Constraints
        self._set_constraints()

        # Solving
        self.model.optimize()

        self.runtime = time.time() - start_time

        self.policy = np.zeros((self.env.state_dim, self.env.action_dim))
        for s in range(self.env.state_dim):
            for a in range(self.env.action_dim):
                self.policy[s, a] = self.var[(s, a)].X

        self.value = policy_evaluation(self.env, self.policy, 1.0, 1e-5)
