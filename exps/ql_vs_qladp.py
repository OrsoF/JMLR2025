# Expérience Q_Learning vs Q_learning+adp
# Idée : prendre une trajectoire de n éléments tirés selon epsilon greedy. Prendre la même initialisation ql et ql+adp et comparer la distance à la fin. Comme modèle on va prendre tandem ?
# Problème: ou viens la désagrégation ? Si on prends une désagrégation

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
trajectory_length = 400
n_trajectory = 100
epsilon = 0.1
discount = 0.9

model: GenericModel = import_models_from_file(model_name)(ss, aa)
model.create_model()
model._model_to_numpy()
optimal_value = get_exact_value(model, "discounted", discount)


def distance(q_value):
    return np.linalg.norm(q_value.max(axis=1) - optimal_value, ord=np.inf)


def q_learning_update(
    current_q: np.ndarray,
    learning_rate: float,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    discount: float,
):
    current_q[state, action] += learning_rate * (
        reward + discount * current_q[next_state].max() - current_q[state, action]
    )
    return current_q


def epsilon_greedy_policy(
    current_q: np.ndarray, current_state: int, epsilon: float, action_dim: int
):
    if np.random.random() < epsilon:
        return np.random.randint(action_dim)
    else:
        return np.argmax(current_q[current_state])


def improvement_ql(n_exp: int = 5):
    imp = 0
    for _ in trange(n_exp):
        q_value = np.zeros((model.state_dim, model.action_dim))
        # print(distance(np.zeros((model.state_dim, model.action_dim))))
        for _ in range(n_trajectory):
            state = np.random.randint(model.state_dim)
            for _ in range(trajectory_length):
                action = epsilon_greedy_policy(
                    q_value, state, epsilon, model.action_dim
                )
                next_state, reward = generate_sars(model, state, action)
                q_value = q_learning_update(
                    q_value, learning_rate, state, action, reward, next_state, discount
                )
                state = next_state
        imp += (
            distance(np.zeros((model.state_dim, model.action_dim))) - distance(q_value)
        ) / np.linalg.norm(optimal_value, ord=np.inf)
    return imp / n_exp


print(1 - improvement_ql())


def improvement_qladp(n_exp: int = 5):
    imp = 0
    previously_seen_states = [0 for _ in range(5)]
    for _ in trange(n_exp):
        q_value = np.zeros((model.state_dim, model.action_dim))
        # print(distance(np.zeros((model.state_dim, model.action_dim))))
        for _ in range(2 * n_trajectory):
            state = np.random.randint(model.state_dim)
            for _ in range(2 * trajectory_length // 5):
                action = epsilon_greedy_policy(
                    q_value, state, epsilon, model.action_dim
                )
                next_state, reward = generate_sars(model, state, action)
                previously_seen_states[
                    np.random.randint(len(previously_seen_states))
                ] = state
                for current_state in previously_seen_states:
                    q_value = q_learning_update(
                        q_value,
                        learning_rate,
                        current_state,
                        action,
                        reward,
                        next_state,
                        discount,
                    )
                state = next_state

        imp += (
            distance(np.zeros((model.state_dim, model.action_dim))) - distance(q_value)
        ) / np.linalg.norm(optimal_value, ord=np.inf)
    return imp / n_exp


print(1 - improvement_qladp())
