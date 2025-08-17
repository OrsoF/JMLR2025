from solvers.aggregated_qvi import Solver as Aqvi
import numpy as np
from models.tandem import Model, param_list
from utils.exact_value_function import get_exact_value, distance_to_optimal
from utils.calculus import norminf, bellman_no_max
from time import time
from utils.calculus import optimal_bellman_operator, norminf

CRITERION = "discounted"

level = 2
discount = 0.9
epsilon = 1e-2

model = Model(param_list[level])
model.create_model()
# model._model_to_numpy()

NUMPY, SPARSE = "numpy", "sparse"

sol = Aqvi(model, discount, epsilon, verbose=False, bellman_updates=10, mode=SPARSE)
sol.run()

# distance = distance_to_optimal(avi.value, model, "discounted", discount)
distance = norminf(optimal_bellman_operator(model, sol.value, discount) - sol.value) / (
    1 - discount
)

print("State dim : {}".format(model.state_dim))
print("Final region count : {}".format(sol.partition._number_of_regions()))
print("Distance to optimal : {}".format(distance))
print("Runtime : {}".format(sol.runtime))
