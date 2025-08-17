from utils.data_management import solve
from utils.exact_value_function import get_exact_value
import matplotlib.pyplot as plt
import numpy as np

discount = 0.5
state = 100
action = 10
precision = 1e-2

model, solver = solve("tandem", "aggregated_ql", state, action, discount, precision)

errors = solver.infos["error_to_optimal"]
regions = solver.infos["number_of_regions"]

plt.plot(regions, label="Number of regions")
plt.xlabel("Iterations")
plt.ylabel("Value") 
plt.title("Aggregated Q-Learning")
plt.legend()
plt.show()

plt.plot(errors, label="Error to optimal")
plt.yscale("log")
plt.xlabel("Iterations")
plt.ylabel("Value") 
plt.title("Aggregated Q-Learning")
plt.legend()
plt.show()