from gurobipy import LinExpr, GRB, Model
import numpy as np
from utils.generic_model import GenericModel
import time
from utils.data_management import gurobi_model_creation


class Solver:
    def __init__(self, env: GenericModel, final_precision: float):
        self.env = env
        self.name = "Gurobi_Average_Dual"
        self.transition_matrix_is_sparse: bool = not isinstance(
            self.env.transition_matrix, np.ndarray
        )

    def _create_variables(self):
        self.var = {}
        for s in range(self.env.state_dim):
            for a in range(self.env.action_dim):
                self.var[(s, a)] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
        self.model.update()

    def _define_objective(self):
        self.obj = LinExpr()
        for s in range(self.env.state_dim):
            for a in range(self.env.action_dim):
                self.obj += self.env.reward_matrix[s, a] * self.var[(s, a)]
        self.model.setObjective(self.obj, GRB.MINIMIZE)
        self.model.update()

    def _set_constraints(self):
        for ss in range(self.env.state_dim):
            sum_1 = sum(self.var[(ss, aa)] for aa in range(self.env.action_dim))

            sum_2 = sum(
                self.var[(ss, aa)] * self.env.transition_matrix[aa][ss2, ss]
                for aa in range(self.env.action_dim)
                for ss2 in range(self.env.state_dim)
            )

            self.model.addConstr(sum_1 - sum_2 == 0)

        sum_3 = sum(
            self.var[(ss, aa)]
            for aa in range(self.env.action_dim)
            for ss in range(self.env.state_dim)
        )

        self.model.addConstr(sum_3 == 1)

        self.model.update()

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

        self.q_value = np.zeros((self.env.state_dim, self.env.action_dim))
        for s in range(self.env.state_dim):
            for a in range(self.env.action_dim):
                self.q_value[s, a] = self.var[(s, a)].X
        self.value = self.q_value.max(axis=1)
        self.policy = None

        self.cost = self.model.getObjective().getValue()