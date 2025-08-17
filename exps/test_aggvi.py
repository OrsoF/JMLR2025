from solvers.aggregated_qpim import Solver as Aqpim
from utils.exact_value_function import distance_to_optimal
from utils.calculus import optimal_bellman_operator, norminf

import numpy as np
from models.rooms import Model

discount = 0.5
epsilon = 1e-1

model = Model(100, 4)
model.create_model()

aqpim = Aqpim(model, discount, epsilon)
aqpim.run()
# distance = distance_to_optimal(avi.value, model, "discounted", discount)
distance = norminf(optimal_bellman_operator(model, aqpim.value, discount) - aqpim.value) / (
    1 - discount
)


print("State dim : {}".format(model.state_dim))
print("Final region count : {}".format(aqpim.partition._number_of_regions()))
print("Distance to optimal : {}".format(distance))
print("Runtime : {}".format(aqpim.runtime))
