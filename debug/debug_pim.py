from solvers.aggregated_qvi import Solver as Avi
import numpy as np
from models.rooms import Model, param_list
from utils.exact_value_function import get_exact_value, distance_to_optimal
from utils.calculus import norminf

CRITERION = "discounted"

level = 3
discount = 0.001
epsilon = 1e-1

model = Model(param_list[level])
model.create_model()

from solvers.personal_pim import Solver

sol = Solver(model, discount, epsilon, mode="sparse")
sol.run()
print("PIM runtime : {}".format(sol.runtime))
distance = distance_to_optimal(
    sol.value,
    model,
    CRITERION,
    discount,
)
print("Distance to optimal : {}".format(distance))
