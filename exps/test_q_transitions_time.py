# Here, we compare np vs sparse transition generation time.

from models.mountain import Model
import numpy as np
import time

state_dim = 100000
action_dim = 10
n_exp = int(1e3)

model = Model(state_dim, action_dim)
model.create_model()
# model._normalize_transition_matrix()

rnd_gen = np.random.default_rng(0)


def generate_sars_sparse(model, state: int, action: int) -> tuple[int, float]:
    transitions = model.transition_matrix[action].getrow(state)
    rnd = rnd_gen.random()
    cum_sum = 0.0
    for index, value in zip(transitions.indices, transitions.data):
        cum_sum += value
        if rnd < cum_sum:
            next_state = index
            break
    # next_state = rnd_gen.choice(transitions.indices, p=transitions.data)
    # cdf = np.cumsum(transitions.data)
    # next_state = np.searchsorted(cdf, rnd_gen.random())
    reward = model.reward_matrix[state, action]
    return next_state, reward


def generate_sars_numpy(model, state: int, action: int) -> tuple[int, float]:
    transitions = model.transition_matrix[action, state, :]
    next_state = rnd_gen.choice(model.state_dim, p=transitions)
    reward = model.reward_matrix[state, action]
    return next_state, reward


def test_sparse_generation_time(n_exp):
    start_time = time.time()
    for _ in range(n_exp):
        state = rnd_gen.integers(model.state_dim)
        action = rnd_gen.integers(model.action_dim)
        state, reward = generate_sars_sparse(model, state, action)

    end_time = time.time()
    sparse_time = end_time - start_time
    print(f"Sparse generation time: {100000* sparse_time/n_exp:.4f} seconds")


def test_numpy_generation_time(n_exp):
    model._model_to_numpy()
    start_time = time.time()
    for _ in range(n_exp):
        state = rnd_gen.integers(model.state_dim)
        action = rnd_gen.integers(model.action_dim)
        state, reward = generate_sars_numpy(model, state, action)

    end_time = time.time()
    sparse_time = end_time - start_time
    print(f"Numpy generation time: {100000* sparse_time/n_exp:.4f} seconds")


test_sparse_generation_time(n_exp)
test_numpy_generation_time(n_exp)
