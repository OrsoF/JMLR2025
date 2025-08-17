from solvers.aggregated_vi import Solver as Avi
import numpy as np
from models.rooms import Model, param_list
from time import time
from utils.calculus_projected import projected_optimal_bellman_operator
from utils.calculus import optimal_bellman_operator
from scipy.sparse import csr_matrix

level = 0
discount = 0.9
epsilon = 1e-3

env = Model(param_list[level])
env.create_model()

print(env.state_dim, discount, epsilon)

avi = Avi(env, discount, epsilon)
avi.run()

start_time = time()
value = csr_matrix(np.zeros((len(avi.partition.states_in_region))))
for _ in range(100000):
    value = projected_optimal_bellman_operator(
        env,
        discount,
        value,
        avi.partition.aggregate_transition_matrix,
        avi.partition.aggregate_reward_matrix,
        avi.partition.weights,
    )
print(time() - start_time)

start_time = time()
value = np.zeros((env.state_dim))
for _ in range(100000):
    value = optimal_bellman_operator(env, value, discount)
print(time() - start_time)
