from solvers.aggregated_vi import Solver as Avi
from solvers.aggregated_qvi import Solver as Aqvi
from solvers.aggregated_pim import Solver as Apim

from solvers.personal_vi import Solver as Vi
from solvers.personal_pim import Solver as Pim
from solvers.chen_td import Solver as Chen
from solvers.bertsekas_pi import Solver as Bert


from utils.generic_model import NUMPY, SPARSE


import numpy as np

level = 1
discount = 0.9
epsilon = 1e-1
mode = NUMPY

from models.dam import Model, param_list

env = Model(param_list[level])
env.create_model(mode)

agg_solvers = [Avi, Aqvi, Apim]
dp_solvers = [Vi, Pim, Chen, Bert]

agg_times = []
for Solver in agg_solvers:
    solver = Solver(env, discount, epsilon)
    solver.run()
    agg_times.append(solver.runtime)
    print("{} : {}".format(solver.name, np.round(solver.runtime, 3)))

agg_time = min(agg_times)

dp_times = []
for Solver in dp_solvers:
    solver = Solver(env, discount, epsilon)
    solver.run()
    dp_times.append(solver.runtime)
    print("{} : {}".format(solver.name, np.round(solver.runtime, 3)))

dp_time = min(dp_times)

print()
print("Model : {}, discount = {}, S = {}".format(env.name, discount, env.state_dim))
print()
score = np.round((agg_time + 1e-5) / (dp_time + 1e-5), 3)
print("Agg/DP : {} < 1".format(score))
print()
print(dp_solvers[np.argmin(dp_times)](env, discount, epsilon).name)
print(agg_solvers[np.argmin(agg_times)](env, discount, epsilon).name)
