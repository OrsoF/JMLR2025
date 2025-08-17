import numpy as np
import os
import argparse

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

assert args.model is not None
assert args.state is not None
assert args.precision is not None
action_dim = 4

args.discount = 1.0

solver_list = [
    "aggregated_vi_total",
    "aggregated_qvi_total",
    "personal_vi",
    "chen_td",
    "aggregated_pim_total",
    "personal_pim",
    # "bertsekas_pi",
]

solver_name_list = [
    import_solver_from_file(solver_name).name for solver_name in solver_list
]
filename = os.path.join(
    "results",
    "agg_{}_{}_{}_{}.txt".format(args.model, args.state, action_dim, args.discount),
)
experience_presentation = (
    "Model : {}, Statedim : {}, Actiondim : {}, Discount : {}".format(
        args.model, args.state, action_dim, args.discount
    )
)
write_text(filename, experience_presentation)

headline = "$\statedim$ & " + " & ".join(solver_name_list) + " \\\\"
write_text(filename, headline, show=True)


latex_line = "{}".format(args.state)
for solver_name in solver_list:
    model, solver = solve(
        args.model, solver_name, args.state, action_dim, args.discount, args.precision
    )
    latex_line += " & {}".format(np.round(solver.runtime, 1))
    write_text(filename, latex_line, show=True)

latex_line += " \\\\"
write_text(filename, latex_line, show=True)
