import sys
from IPython.core import ultratb
from gurobipy import norm
import numpy as np

sys.excepthook = ultratb.FormattedTB(color_scheme="Linux", call_pdb=False)

from utils.data_management import solve
from utils.calculus import norminf

model, solver = solve("mountain", "mdptoolbox_vi", 900, 10, 0.99, 1e-1)

v = solver.value

n = int(len(v) ** 0.5)

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(
    np.reshape(v, (n, n)), cmap="YlGnBu", xticklabels=5, yticklabels=5
)
plt.title("Optimal value function")
plt.xlabel("Position")
plt.ylabel("Velocity")
plt.show()
