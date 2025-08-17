# Question: dans Q-learning, est-ce que les updates sont à peu près uniformes ?

import numpy as np
import matplotlib.pyplot as plt
from utils.data_management import import_models_from_file
from utils.generic_model import GenericModel
from utils.calculus import generate_sars
from utils.exact_value_function import distance_to_optimal_q, get_exact_value
from tqdm import trange

model_name = "tandem"
ss = 500
aa = 4
learning_rate = 0.1
trajectory_length = 200
n_trajectory = 1000
epsilon = 0.1
discount = 0.9

model: GenericModel = import_models_from_file(model_name)(ss, aa)
model.create_model()
model._model_to_numpy()
optimal_value = get_exact_value(model, "discounted", discount)


def distance(q_value):
    return np.linalg.norm(q_value.max(axis=1) - optimal_value, ord=np.inf)


current_data = {}


def q_learning_update(
    current_q: np.ndarray,
    learning_rate: float,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    discount: float,
):
    dt = reward + discount * current_q[next_state].max() - current_q[state, action]
    current_q[state, action] += learning_rate * dt
    if state in current_data:
        current_data[state].append(dt)
    else:
        current_data[state] = [dt]
    return current_q


def epsilon_greedy_policy(
    current_q: np.ndarray, current_state: int, epsilon: float, action_dim: int
):
    if np.random.random() < epsilon:
        return np.random.randint(action_dim)
    else:
        return np.argmax(current_q[current_state])


q_value = np.zeros((model.state_dim, model.action_dim))
# print(distance(np.zeros((model.state_dim, model.action_dim))))
for _ in trange(n_trajectory):
    state = np.random.randint(model.state_dim)
    for _ in range(trajectory_length):
        action = epsilon_greedy_policy(q_value, state, epsilon, model.action_dim)
        next_state, reward = generate_sars(model, state, action)
        q_value = q_learning_update(
            q_value, learning_rate, state, action, reward, next_state, discount
        )
        state = next_state
print(distance(np.zeros((model.state_dim, model.action_dim))) - distance(q_value))


state = np.random.randint(model.state_dim)
while state not in current_data or len(current_data[state]) < 10:
    state = np.random.randint(model.state_dim)

print(state)
plt.hist(current_data[state], bins=50)
plt.show()

# res = []
# for state in range(model.state_dim):
#     if state in current_data:
#         res.append(np.mean(current_data[state]))

# plt.plot(res)
# plt.plot(optimal_value)
# plt.show()
