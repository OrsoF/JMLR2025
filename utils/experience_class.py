import os
from mdptoolbox.mdp import RelativeValueIteration, ValueIteration, PolicyIteration
from typing import List, Tuple, Optional
from utils.generic_model import GenericModel
from utils.generic_solver import GenericSolver
from utils.data_management import (
    import_all_solvers,
    get_models_from_model_file,
)
from utils.calculus import norminf
from utils.exact_value_function import get_exact_value
import numpy as np
import pandas as pd
import time


def model_parameters_to_str(params: dict) -> str:
    str_param = ""
    for key in params.keys():
        if (
            isinstance(params[key], str)
            or isinstance(params[key], float)
            or isinstance(params[key], int)
        ):
            str_param += "_{}_{}".format(key, params[key])
    return str_param


RESULT_PATH = os.path.join(os.getcwd(), "results")
SAVED_MODELS_PATH = os.path.join(os.getcwd(), "saved_models")
EXACT_VALUE_PATH = os.path.join(os.getcwd(), "saved_value_functions")
METHOD = "discounted"


class Experience:
    def __init__(
        self,
        experience_parameters: dict,
    ) -> None:
        self.n_exp = experience_parameters["measure_repetition"]
        self.verbose = experience_parameters["verbose"]

        self.model_list = get_models_from_model_file(experience_parameters["model"])
        self.solver_list = import_all_solvers()
        self.precisions = experience_parameters["precisions"]
        self._try_folder_creation()

        self.discounts = experience_parameters["discounts"]
        self.experience_full_name = experience_parameters["experience_name"]

        self.results = []

    def run(self):
        """
        Run the full experiment.
        """
        for model in self.model_list:
            model: GenericModel
            model.create_model()
            for solver in self.solver_list:
                for discount in self.discounts:
                    for final_precision in self.precisions:

                        solver: GenericSolver
                        solver.__init__(model, discount, final_precision)

                        if self.verbose:
                            print(
                                "{} {} {} {}".format(
                                    model.name, solver.name, discount, final_precision
                                )
                            )

                        self._model_specific_solver_specific_experience(
                            model, solver, discount, final_precision
                        )

            model.lighten_model()

    def _try_folder_creation(self):
        """
        Folder creation to save results,
        models and value functions.
        """
        for folder_path in [
            RESULT_PATH,
            SAVED_MODELS_PATH,
            EXACT_VALUE_PATH,
        ]:
            try:
                os.mkdir(folder_path)
            except FileExistsError:
                pass

    def _model_specific_solver_specific_experience(
        self,
        model: GenericModel,
        solver: GenericSolver,
        discount: float,
        final_precision: float,
    ):
        """
        For the given model and solver,
        add the experience information to self.results.
        """
        assert hasattr(model, "transition_matrix") and hasattr(
            model, "reward_matrix"
        ), "Model should have been created using model.create_model()."

        # As transition and reward have been deleted, we load it back.
        exact_value_function = get_exact_value(model, METHOD, discount)

        for _ in range(self.n_exp):
            solver.__init__(model, discount, final_precision)
            solver.run()

            difference = solver.value - exact_value_function
            difference = difference - difference[0]
            gap_to_optimal = norminf(difference)

            result_instance = {
                "instance_name": model.name,
                "solver_name": solver.name,
                "discount": discount,
                "runtime": solver.runtime,
                "distance_to_optimal": gap_to_optimal,
                "state_dim": model.state_dim,
                "action_dim": model.action_dim,
                "instance_parameters": model_parameters_to_str(model.params),
                "final_precision": final_precision,
                "transition_density": model.get_transition_density(),
                "reward_density": model.get_reward_density(),
            }
            self.results.append(result_instance)

        self._save_current_results()

    def get_current_experience_number(self) -> int:
        if not hasattr(self, "current_exp_number"):
            self.current_exp_number = len(
                os.listdir(os.path.join(os.getcwd(), "results"))
            )
        return self.current_exp_number

    def _save_current_results(self):
        """
        Save the current self.results list to an excel file.
        """
        results_dataframe = pd.DataFrame(self.results)
        file_name = "{}_experience_{}.csv".format(
            self.get_current_experience_number(), self.experience_full_name
        )
        results_dataframe.to_csv(os.path.join(RESULT_PATH, file_name))
