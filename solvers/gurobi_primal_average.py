from gurobipy import LinExpr, GRB, Model
import numpy as np
from utils.generic_model import GenericModel
from gurobipy import Model
import time


def gurobi_model_creation() -> Model:
    model = Model("MDP")
    model.setParam("OutputFlag", 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam("LogToConsole", 0)
    return model


class Solver:
    def __init__(self, env: GenericModel, final_precision: float):
        self.env = env
        self.name = "Gurobi_Average_Primal"
        self.transition_matrix_is_sparse: bool = not isinstance(
            self.env.transition_matrix, np.ndarray
        )

    def _create_variables(self):
        self.var = {}
        self.var["g"] = self.model.addVar(vtype=GRB.CONTINUOUS)
        for s in range(self.env.state_dim):
            self.var[s] = self.model.addVar(vtype=GRB.CONTINUOUS)
        self.model.update()

    def _define_objective(self):
        self.obj = LinExpr()
        self.obj += self.var["g"]
        self.model.setObjective(self.obj, GRB.MINIMIZE)

    def _set_constraints(self):
        for s in range(self.env.state_dim):
            for a in range(self.env.action_dim):
                sum_1 = self.var["g"] + self.var[(s)]

                sum_2 = sum(
                    self.env.transition_matrix[a][s, sp] * self.var[sp]
                    for sp in range(self.env.state_dim)
                )

                sum_3 = self.env.reward_matrix[s, a]

                self.model.addConstr(sum_1 - sum_2 >= sum_3, "Contrainte%d" % s)

    def run(self):
        start_time = time.time()
        self.model: Model = gurobi_model_creation()

        # Variables
        self._create_variables()

        # Objective
        self._define_objective()

        # Constraints
        self._set_constraints()

        self.model.optimize()

        self.runtime = time.time() - start_time

        self.value = np.zeros((self.env.state_dim))
        for s in range(self.env.state_dim):
            self.value[s] = self.var[(s)].X
        self.policy = None

        self.cost = self.model.getObjective().getValue()