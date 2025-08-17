from ast import arg
from json import load
from tracemalloc import start
from arrow import get
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import trange
from regex import F
from solvers.aggregated_ql import Solver as agg_Solver
from solvers.personal_ql import Solver as ql_Solver
from solvers.reynolds_ql import Solver as rey_Solver
from utils.exact_value_function import distance_to_optimal
from utils.data_management import load_and_build_model, solve
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument(
    "--model",
    type=str,
)
from datetime import datetime
import os
from params import count_of_experiments, discount, state_dim

model_name = argparse.parse_args().model

start_time = time.time()

################### Aggregated Q-Learning ###################

<<<<<<< HEAD
count_of_experiments = 2

learning_rate = args.lr
decay_rate = args.decay

best_action = 0.95

######## Aggregated Q-Learning #########

errors = []
for i in trange(count_of_experiments):
    agg_solver = agg_Solver(
        model=model,
        discount=discount,
        final_precision=0.01,
        total_learning_step=total_learning_step,
        traj_length=traj_len,
        n_traj=n_traj,
        leaning_rate=learning_rate,
        decay_rate=decay_rate,
        best_action=best_action,
        epsilon_division=0.001,
        prop_agg_step=0.5,
        get_infos=True,
    )
    agg_solver.run()
    errors.append(agg_solver.infos["error_to_optimal"])

errors = np.array(errors)
average = errors.mean(axis=0)
std = errors.std(axis=0)
=======
errors_agg_ql = []
for _ in trange(count_of_experiments):
    model, solver = solve(model_name, "aggregated_ql", state_dim, 4, discount, 0.1)
    errors_agg_ql.append(solver.infos["error_to_optimal"])
errors_agg_ql = np.array(errors_agg_ql)
average_agg_ql = errors_agg_ql.mean(axis=0)
std_agg_ql = errors_agg_ql.std(axis=0)
>>>>>>> 0f743aa757c81cf726cec23b6dd4a892486554db

color = "blue"
plt.plot(average_agg_ql, label="Aggregated Q-Learning", color=color)
plt.fill_between(
    range(len(average_agg_ql)),
    average_agg_ql - std_agg_ql,
    average_agg_ql + std_agg_ql,
    alpha=0.2,
    color=color,
)

plt.xlabel("Episode")
plt.ylabel(r"$\|Q - Q^*\|$")
plt.yscale("log")
plt.grid()
plt.legend()

################### Q-Learning ###################

errors_ql = []
for _ in trange(count_of_experiments):
    model, solver = solve(model_name, "personal_ql", state_dim, 4, discount, 0.1)
    errors_ql.append(solver.infos["error_to_optimal"])
errors_ql = np.array(errors_ql)
average_ql = errors_ql.mean(axis=0)
std_ql = errors_ql.std(axis=0)


color = "orange"
plt.plot(average_ql, label="Q-Learning", color=color)
plt.fill_between(
    range(len(average_ql)),
    average_ql - std_ql,
    average_ql + std_ql,
    alpha=0.2,
    color=color,
)

plt.xlabel("Episode")
plt.ylabel(r"$\|Q - Q^*\|$")
plt.yscale("log")
plt.grid()
plt.legend()

################ Reynolds QVI ###################

errors_reynolds = []
for _ in trange(count_of_experiments):
    model, solver = solve(model_name, "reynolds_ql", state_dim, 4, discount, 0.1)
    errors_reynolds.append(solver.infos["error_to_optimal"])
errors_reynolds = np.array(errors_reynolds)
average_reynolds = errors_reynolds.mean(axis=0)
std_reynolds = errors_reynolds.std(axis=0)

color = "green"
plt.plot(average_reynolds, label="Reynolds", color=color)
plt.fill_between(
    range(len(average_reynolds)),
    average_reynolds - std_reynolds,
    average_reynolds + std_reynolds,
    alpha=0.2,
    color=color,
)

plt.xlabel("Episode")
plt.ylabel(r"$\|Q - Q^*\|$")
plt.yscale("log")
plt.grid()
plt.legend()

FIG_NAME = (
    f"agg_vs_base_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{model_name}.pdf"
)
plt.savefig(
    f"figures/{FIG_NAME}",
    bbox_inches="tight",
)

print('Total time:', time.time() - start_time)

plt.ylim(bottom = 1e-2)

plt.show()


