from utils.data_management import load_and_build_model
from utils.exact_value_function import get_exact_value
from solvers.aggregated_ql import Solver
import matplotlib.pyplot as plt
import numpy as np

discount = 0.9
state = 100
action = 10
model = load_and_build_model("rooms", state, action)
agg_ql = Solver(
    model=model,
    discount=discount,
    final_precision=0.1,
    get_infos=True,
    epsilon_division=1.0,
    total_learning_step=1000,
    offset=10,
    leaning_rate=1.0,
    traj_length=10,
)

agg_ql.run()

regions_count = agg_ql.infos["number_of_regions"]
errors = agg_ql.infos["error_to_optimal"]

plt.plot(regions_count, label="Number of regions")
plt.xlabel("Learning steps")
plt.ylabel("Value")
plt.show()

plt.plot(errors, label="Error to optimal")
plt.xlabel("Learning steps")
plt.ylabel("Value")
plt.show()

plt.plot(agg_ql.value)
plt.xlabel("State")
plt.ylabel("Value")
plt.title("Value function")
plt.show()
