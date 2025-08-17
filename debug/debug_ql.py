from solvers.personal_ql import Solver
import numpy as np
from models.unichain import Model, param_list
from utils.exact_value_function import get_exact_value, distance_to_optimal
from utils.calculus import norminf
import matplotlib.pyplot as plt

CRITERION = "discounted"

level = 0
discount = 0.99
epsilon = 1e-1

model = Model(param_list[level])
model.create_model()
model._model_to_numpy()
model.reward_matrix -= 1

sol = Solver(model, discount)
sol.run()
print("Q-learning runtime : {}".format(np.round(sol.runtime, 3)))
distance = distance_to_optimal(
    sol.value,
    model,
    CRITERION,
    discount,
)
distance = np.round(distance, 3)
print("Distance to optimal : {}".format(distance))

try:
    import seaborn as sns

    sns.heatmap(sol.value.reshape((10, 10)) - sol.value[2])
    plt.show()
except ValueError:
    pass
