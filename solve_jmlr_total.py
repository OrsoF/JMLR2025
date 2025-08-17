from utils.data_management import solve
import argparse

# from reset import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--state", type=int)
parser.add_argument("--action", type=int, required=False)
parser.add_argument("--precision", type=float, required=False)
parser.add_argument("--repeat", type=int, required=False)
args = parser.parse_args()

print(f"Model {args.model}...")

for solver in [
    "aggregated_vi_total",
    "aggregated_qvi_total",
    "personal_vi",
    "aggregated_pim_total",
    "personal_pim",
]:
    args.solver = solver

    assert args.model is not None, "Model must be specified."
    assert args.solver is not None, "Solver must be specified."
    assert args.state is not None, "State dim must be specified. --state <int>"
    args.action = 10 if args.action is None else args.action
    args.precision = 1e-2 if args.precision is None else args.precision
    args.repeat = 1 if args.repeat is None else args.repeat

    model, solver = solve(
        args.model,
        args.solver,
        args.state,
        args.action,
        1.0,
        args.precision,
        args.repeat,
    )
    # print(f"Runtime: {solver.runtime}")
    # print(solver.value)

    # if hasattr(solver, "partition"):
    #     print(f"Regions: {solver.partition._number_of_regions()}")
