from solvers.bertsekas_pi import Solver
import numpy as np
from models.rooms import Model, param_list
from utils.exact_value_function import distance_to_optimal

level = 0
discount = 0.8
epsilon = 1e-2
criterion = "discounted"

model = Model(param_list[level])
model.create_model(mode="sparse")

sol = Solver(model, discount, epsilon)
sol.run()
print("Bertsekas : {}".format(sol.runtime))

distance = distance_to_optimal(
    sol.value,
    model,
    criterion,
    discount,
)
print("Distance to optimal : {}".format(distance))
