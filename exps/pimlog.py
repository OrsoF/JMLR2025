from utils.data_management import solve
from utils.generic_model import GenericModel
from utils.calculus import bellman_no_max
from utils.exact_value_function import distance_to_optimal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


model, _ = solve("swim", "personal_vi", 100, 10, 0.1, 1e-1)
discount = 0.9999
precision = 1e-2
model._normalize_reward_matrix()


def log_bellman_update(model: GenericModel, discount: float, log_value: np.ndarray):
    return np.log(np.max(bellman_no_max(model, np.exp(log_value), discount), axis=1))


results = [[], [], []]

alpha = 0.1
alphas = np.linspace(-alpha, alpha, 100)
A = 10
B = 40

for alpha in tqdm(alphas):

    # log_value = np.zeros((model.state_dim))
    # for _ in range(A):
    #     for _ in range(B):
    #         next_log_value = log_bellman_update(model, discount, log_value)
    #         log_value = (1 + alpha) * (next_log_value - log_value) + log_value

    # results[0].append(
    #     distance_to_optimal(np.exp(log_value), model, "discounted", discount)
    # )

    value = np.zeros((model.state_dim))
    for _ in range(A):
        for _ in range(B):
            next_value = bellman_no_max(model, value, discount).max(axis=1)
            value = (1 + alpha) * (next_value - value) + value

    results[1].append(distance_to_optimal(value, model, "discounted", discount))

    # value = np.zeros((model.state_dim))
    # for _ in range(10):
    #     for _ in range(10):
    #         next_value = bellman_no_max(model, value, discount).max(axis=1)
    #         value = (1) * (next_value - value) + value

    # results[2].append(distance_to_optimal(value, model, "discounted", discount))

# plt.plot(alphas, results[0], label="log")
plt.plot(alphas, results[1], label="val")

plt.yscale("log")
plt.legend()
plt.show()
