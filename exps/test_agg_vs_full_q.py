from solvers.aggregated_qvi import Solver as Aqi
import numpy as np
from models.rooms import Model, param_list
from time import time
from utils.calculus_projected import projected_optimal_q_bellman_operator
from utils.calculus import optimal_bellman_operator

level = 0
discount = 0.9
epsilon = 1e-3

env = Model(param_list[level])
env.create_model()

print(env.state_dim, discount, epsilon)

avi = Aqi(env, discount, epsilon)
avi.run()

start_time = time()
value = np.zeros((len(avi.partition.states_in_region), env.action_dim))
for _ in range(10000):
    value = projected_optimal_q_bellman_operator(
        env,
        discount,
        value,
        avi.partition.aggregate_transition_matrix,
        avi.partition.aggregate_reward_matrix,
    )
print(time() - start_time)


start_time = time()
value = np.zeros((env.state_dim))
for _ in range(10000):
    value = optimal_bellman_operator(env, value, discount)
print(time() - start_time)
