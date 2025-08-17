from models.peg_solitaire import Model
import numpy as np

model = Model(int(1e7), 4)
model.create_model()
print()
print(model.name)
print("State dim : {}".format(model.state_dim))
print("Action dim : {}".format(model.action_dim))
print("Transition density : {}".format(np.round(model.get_transition_density(), 8)))
print("Reward density : {}".format(np.round(model.get_reward_density(), 4)))

# from solvers.personal_vi import Solver

# discount = 0.99
# precision = 1e-2

# solver = Solver(model, discount, precision)
# solver.run()
# print(solver.runtime)
