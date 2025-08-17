from solvers.marmote_vigs import Solver

import numpy as np
from models.rooms import Model, param_list

from utils.calculus import norminf

level = 0
discount = 0.9
epsilon = 1e-3

env = Model(param_list[level])
env.create_model()

marmotevi = Solver(env, discount, epsilon)
marmotevi.run()
