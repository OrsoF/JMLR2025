import numpy as np

from models.unichain import Model, param_list

level = 2
discount = 0.99
precision = 1e-2

env = Model(param_list[level])
env.create_model()

from solvers.mdptoolbox_vi import Solver

sol = Solver(env, discount, precision)
sol.run()
print("MDPToolbox runtime : {}".format(np.round(sol.runtime, 2)))

from solvers.marmote_vigs import Solver

sol = Solver(env, discount, precision)
sol.run()
print("Marmote runtime : {}".format(np.round(sol.runtime, 2)))
