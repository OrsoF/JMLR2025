from utils.data_management import solve
import argparse

# from reset import *


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--state", type=int)
parser.add_argument("--discount", type=float)
parser.add_argument("--action", type=int, required=False)
parser.add_argument("--precision", type=float, required=False)
parser.add_argument("--repeat", type=int, required=False)
args = parser.parse_args()

import sys

import sys
from types import ModuleType, FunctionType
from gc import get_referents

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError("getsize() does not take argument of type: " + str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


print(f"Model {args.model}...")

for solver in [
    "aggregated_vi",
    "aggregated_qvi",
    "personal_vi",
    "aggregated_pim",
    "personal_pim",
    "bertsekas_pi",
]:
    print("Solver:", solver)
    args.solver = solver

    assert args.model is not None, "Model must be specified."
    assert args.solver is not None, "Solver must be specified."
    assert args.state is not None, "State dim must be specified. --state <int>"
    assert (
        args.discount is not None
    ), "Discount factor must be specified., --discount <float>"
    args.action = 10 if args.action is None else args.action
    args.precision = 1e-1 if args.precision is None else args.precision
    args.repeat = 1 if args.repeat is None else args.repeat

    model, solver = solve(
        args.model,
        args.solver,
        args.state,
        args.action,
        args.discount,
        args.precision,
        args.repeat,
    )
    # print(f"Runtime: {solver.runtime}")
    # print(solver.value)
