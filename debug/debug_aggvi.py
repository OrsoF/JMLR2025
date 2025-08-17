from solvers.aggregated_vi import Solver as Avi
import numpy as np
from models.tandem import Model, param_list
from utils.exact_value_function import distance_to_optimal
from utils.calculus import norminf, optimal_bellman_operator

level = 2
discount = 0.9
epsilon = 1e-2
criterion = "discounted"
mode = "sparse"

model = Model(param_list[level])
model.create_model()

sol = Avi(model, discount, epsilon, mode=mode)
sol.run()

# distance = distance_to_optimal(avi.value, model, "discounted", discount)
distance = norminf(optimal_bellman_operator(model, sol.value, discount) - sol.value) / (
    1 - discount
)
print("State dim : {}".format(model.state_dim))
print("Final region count : {}".format(sol.partition._number_of_regions()))
print("Distance to optimal : {}".format(distance))
print("Runtime : {}".format(sol.runtime))
