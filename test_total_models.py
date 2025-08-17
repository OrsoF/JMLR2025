from models.taxi_total import Model
import numpy as np

state_dim = 10000
action_dim = 6

model = Model(state_dim, action_dim)
model.create_model()

print(model.name)
print("State dim : {}".format(model.state_dim))
print("Action dim : {}".format(model.action_dim))
print("Transition density : {}".format(np.round(model.get_transition_density(), 8)))
print("Reward density : {}".format(np.round(model.get_reward_density(), 4)))
print()

discount = 1.0
precision = 1e-2

from solvers.mdptoolbox_vi_total import Solver

solver = Solver(model, discount, precision)
solver.run()
print(f"Runtime: {solver.runtime}")

print()

from solvers.mdptoolbox_pim_total import Solver

solver = Solver(model, discount, precision)
solver.run()
print(f"Runtime: {solver.runtime}")

print(solver.value)