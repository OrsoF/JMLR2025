from cProfile import label
from utils.calculus import optimal_bellman_operator
from utils.generic_model import GenericModel
import numpy as np
from typing import Optional
from utils.exact_value_function import distance_to_optimal, get_exact_value
import matplotlib.pyplot as plt


class GenericSolver:
    def __init__(
        self, model: GenericModel, discount: float, final_precision: float
    ) -> None:
        self.name: str
        self.model = model
        self.discount = discount

        self.value: np.ndarray
        self.policy: np.ndarray
        self.runtime: float

    def run(self):
        # Example
        pass

    def build_solution(self):
        # Example
        pass

    def distance_to_optimal(self):
        return distance_to_optimal(self.value, self.model, "discounted", self.discount)

    def bellman_residual(self, ord=np.inf) -> float:
        bellman_value = optimal_bellman_operator(self.model, self.value, self.discount)
        return np.linalg.norm(self.value - bellman_value, ord=ord)

    def plot_value_function(self):
        """Plot the final value function found by the solver."""
        optimal_value = get_exact_value(self.model, "discounted", self.discount)
        plt.plot(self.value, label="V")
        plt.plot(optimal_value, label="V*")
        plt.xlabel("State")
        plt.ylabel("Value")
        plt.legend()
        plt.show()
