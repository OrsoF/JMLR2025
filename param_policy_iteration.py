from models.rooms import Model

model = Model(100, 4)
model.create_model()

import numpy as np
from solvers.personal_pim import Solver as PIM

solver = PIM(model, 0.9, 0.01)
solver.run()

print(solver.runtime)