from solvers.aggregated_qvi import Solver as Aqvi
from models.cliff import Model as Cliff

model = Cliff(48, 10)
model.create_model()

solver = Aqvi(model, 0.99, 1e-1)
solver.run()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

space = np.zeros((12, 4)).T
for i, region in enumerate(solver.partition.states_in_region):
    for state in region:
        x, y = state // 12, state % 12
        space[x, y] = i

space = np.flip(space, axis=0)
sns.heatmap(space, annot=True, cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)
plt.show()
