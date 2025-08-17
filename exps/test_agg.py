from solvers.aggregated_vi_lasting import Solver as Avi
from solvers.personal_vi import Solver as Vi
import numpy as np
from models.rooms import Model, param_list

level = 2
discount = 0.99
epsilon = 1e-2

env = Model(param_list[level])
env.create_model()

print(env.state_dim, discount, epsilon)

vi = Vi(env, discount, epsilon)
vi.run()
# print("Value Iteration : {}".format(vi.runtime))


sol = Avi(env, discount, epsilon)
sol.run()
# print("AggVI : {}".format(sol.runtime))

print("AggVI / VI : {} < 1".format(sol.runtime / vi.runtime))
