from cProfile import label
from utils.data_management import solve
from utils.exact_value_function import get_exact_value
import matplotlib.pyplot as plt

discount = 0.9
state = 100
action = 4
precision = 1e-2

model, solver = solve("tandem", "reynolds_qvi", state, action, discount, precision)

plt.plot(solver.infos["number_of_regions"], label="reynolds")

model, solver = solve("tandem", "aggregated_ql", state, action, discount, precision)

plt.plot(solver.infos["number_of_regions"], label="aggql")
plt.xlabel("Iteration")
plt.ylabel("Number of regions")
plt.title("Number of regions over iterations")
plt.show()
