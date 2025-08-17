import numpy as np
import argparse
import os

from utils.data_management import import_solver_from_file
from utils.data_management import write_text, solve

try:
    os.mkdir("results")
except FileExistsError:
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--state", type=int)
parser.add_argument("--precision", type=float)
args = parser.parse_args()

action_dim = 10

solver_list = [
    # "marmote_vi_total",
    # "marmote_vigs_total",
    # "mdptoolbox_vi_total",
    # "marmote_pim_total",
    # "marmote_pimgs_total",
    # "mdptoolbox_pim_total",
    "gurobi_primal_total",
    "gurobi_dual_total",
]

solver_name_list = [
    import_solver_from_file(solver_name).name for solver_name in solver_list
]
filename = os.path.join(
    "results",
    "marmote_{}_{}_{}.txt".format(args.model, args.state, action_dim),
)

print(
    "Model : {}, Statedim : {}, Actiondim : {}".format(
        args.model, args.state, action_dim
    )
)
print()

headline = "$\statedim$ & " + " & ".join(solver_name_list) + " \\\\"
write_text(filename, headline, show=True)
latex_line = "{}".format(args.state)

for solver_name in solver_list:
    if "gurobi" in solver_name and args.state > 25000:
        continue
    model, solver = solve(
        args.model, solver_name, args.state, action_dim, 1.0, args.precision
    )
    latex_line += " & {}".format(np.round(solver.runtime, 1))
    write_text(filename, latex_line, show=True)

latex_line += " \\\\"
write_text(filename, latex_line, show=True)
