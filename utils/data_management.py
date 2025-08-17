import os
import platform
import shutil
import gurobipy as grb
from importlib.util import spec_from_file_location, module_from_spec
import time
import matplotlib.pyplot as plt
from gurobipy import GRB, Model
import numpy as np
from tqdm import trange

from utils.generic_model import GenericModel

from utils.generic_solver import GenericSolver
from typing import Generic, List, Tuple
from utils.calculus import optimal_bellman_residual

SOLVER_PATH = os.path.join(os.getcwd(), "solvers")
MODEL_PATH = os.path.join(os.getcwd(), "models")
TOY_DISCOUNT = 0.8
TOY_PRECISION = 1e-2


def import_solver_from_file(file_name: str) -> GenericSolver:
    """
    Given a file (as "q_learning.py")
    returns the associated_solver.
    """
    if not file_name.endswith(".py"):
        file_name += ".py"
    spec = spec_from_file_location(
        "__temp_module__", os.path.join(SOLVER_PATH, file_name)
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Solver(ToyModel(0, 0), TOY_DISCOUNT, TOY_PRECISION)


def import_models_from_file(file_name: str):
    if not file_name.endswith(".py"):
        file_name += ".py"
    spec = spec_from_file_location(
        "__temp_module__", os.path.join(MODEL_PATH, file_name)
    )
    assert spec is not None, "File not found {}".format(file_name)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


def get_models_from_model_file(model_file: str) -> list[GenericModel]:
    model, param_list = import_models_from_file(file_name=model_file)
    return [model(params) for params in param_list]


def reset():
    """
    Function that reset the experience environment.
    """
    for folder in ["saved_models", "saved_value_functions", "results", "images"]:
        try:
            shutil.rmtree(os.path.join(os.getcwd(), folder))
        except FileNotFoundError:
            pass


def gurobi_license() -> bool:
    """
    To check if the Gurobi license exists or not.
    Output : True if license, False otherwise.
    """
    gurobi_log_string = "gurobi.log"
    grb.Env(gurobi_log_string)
    f = open(gurobi_log_string, "r")
    log = f.read()
    f.close()
    os.remove(os.path.join(os.getcwd(), gurobi_log_string))
    return "Restricted" not in log


def linux() -> bool:
    """
    To check if the system runs on Linux.
    Output : True if Linux, False otherwise
    """
    return platform.system() == "Linux"


def remove_unused_solver_files(solver_name_list: list) -> list:
    for file_name in ["__pycache__", "_pyMarmoteMDP.so", "pyMarmoteMDP.py"]:
        try:
            solver_name_list.remove(file_name)
        except ValueError:
            pass
    return solver_name_list


def solve(
    model_name: str,
    solver_name: str,
    state_dim: int,
    action_dim: int,
    discount: float,
    precision: float,
    repeat: int = 1,
) -> Tuple[GenericModel, GenericSolver]:
    """Solve a given model with a specified solver."""
    model = import_models_from_file(model_name)
    model: GenericModel = model(state_dim, action_dim)
    model.create_model()
    solver = import_solver_from_file(solver_name)
    solver.__init__(model, discount, precision)
    runtimes = []
    repeat += 1
    if repeat == 1:
        solver.run()
        # print("Runtime: {:.3f} seconds".format(solver.runtime))
    else:
        for run in range(repeat):
            solver.run()
            if run == 0:
                continue
            runtimes.append(solver.runtime)

        runtimes = runtimes[1:]
        mean = np.round(np.mean(runtimes), 1)
        std = np.round(np.std(runtimes), 1)

        print(f"{mean} $\pm$ {std}")
    return model, solver


def load_and_build_model(
    model_name: str, state_dim: int, action_dim: int
) -> GenericModel:
    model = import_models_from_file(model_name)
    model: GenericModel = model(state_dim, action_dim)
    model.create_model()
    return model


def write_text(filename: str, text: str, show: bool = False):
    if show:
        print(text)
    with open(filename, "a") as file:
        # Add a newline before the text if the file already has content
        if file.tell() > 0:
            file.write("\n")
        file.write(text)


def plot_optimal_value(model_name: str, state_dim: int, discount: float = 0.99):
    _, solver = solve(model_name, "marmote_vi", state_dim, 10, discount, 1e-6)
    optimal_value = solver.value
    plt.plot(optimal_value, label="Optimal value of {} model".format(model_name))
    plt.xlabel("State index")
    plt.legend()
    plt.show()


def gurobi_model_creation() -> Model:
    model = Model("MDP")
    model.setParam("OutputFlag", 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam("LogToConsole", 0)
    model.setParam("MemLimit", 16)
    return model


class ToyModel(GenericModel):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.name = "toy_model"
        self.state_dim = 1
        self.action_dim = 1
        self.transition_matrix = [np.ones((1, 1))]
        self.reward_matrix = np.zeros((1, 1))
