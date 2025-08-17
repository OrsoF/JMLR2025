from solvers.aggregated_vi import Solver as Avi
from solvers.aggregated_qvi import Solver as Aqvi
from solvers.aggregated_pim import Solver as Apim

from solvers.personal_vi import Solver as Vi
from solvers.personal_pim import Solver as Pim
from solvers.bertsekas_pi import Solver as Bert
from solvers.chen_td import Solver as Chen

from utils.generic_model import NUMPY, SPARSE
from utils.data_management import import_models_from_file

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import os
import argparse

parser = argparse.ArgumentParser(
    description="Parse command line arguments for plot_discount_impact.py"
)
parser.add_argument("--level", type=int, help="Level")
parser.add_argument("--epsilon", type=float, help="Final precision")
parser.add_argument("--model", type=str, help="Model name")
args = parser.parse_args()

# Access parsed arguments
level = args.level
epsilon = args.epsilon
model_name = args.model
Model, param_list = import_models_from_file(model_name)

# Constants
discounts = [
    0.9,
    0.99,
    0.999,
    0.9999,
]
f = lambda x: np.log10(x) - np.log10(1 - x)
discounts_plot = [f(discount) for discount in discounts]

blues = [
    (0, 0, 1),  # Blue
    (0, 0.5, 1),  # Light Blue
    (0.25, 0.25, 1),  # Lavender Blue
]

oranges = [
    (1, 0.65, 0),  # Orange
    (1, 0.75, 0.25),  # Light Orange
    (1, 0.5, 0),  # Dark Orange
    (1, 0.4, 0),  # Tangerine
]


def plot_and_save_results(results):
    blues = [
        (0, 0, 1),  # Blue
        (0, 0.5, 1),  # Light Blue
        (0.25, 0.25, 1),  # Lavender Blue
    ]

    oranges = [
        (1, 0.65, 0),  # Orange
        (1, 0.75, 0.25),  # Light Orange
        (1, 0.5, 0),  # Dark Orange
        (1, 0.4, 0),  # Tangerine
    ]
    for solver_name, runtimes in results.items():
        if "agg" in solver_name:
            color = blues.pop()
        else:
            color = oranges.pop()
        plt.plot(runtimes, label=solver_name, color=color)
    plt.xticks(range(len(runtimes)), [str(elem) for elem in discounts[: len(runtimes)]])
    plt.xlabel("Discount")
    plt.ylabel("Runtime (s)")
    plt.yscale("log")
    title = "Runtime for {}, precision {}".format(model.name, epsilon)

    plt.title(title)
    plt.legend()
    plt.savefig(file_path, dpi=300)
    plt.cla()


# Model
try:
    model = Model(param_list[level])
except IndexError:
    exit()
model.create_model(mode=SPARSE, check_transition=False)
print(model.name)

file_path = os.path.join(
    "images",
    "discount_impact_{}_{}_{}.png".format(
        model.state_dim, model.name.split("_")[2], np.random.randint(1000)
    ),
)

# Experience

solvers = [Avi, Aqvi, Apim, Vi, Pim, Bert, Chen]
solvers = [Avi, Aqvi, Apim, Vi, Pim]
results = {}

for discount in discounts:
    for Solver in solvers:
        solver = Solver(model, discount, epsilon)
        print("{} {}".format(discount, solver.name))
        solver.run()
        try:
            results[solver.name].append(solver.runtime)
        except KeyError:
            results[solver.name] = [solver.runtime]
    plot_and_save_results(results)
