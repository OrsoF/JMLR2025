from solvers.marmote_vi import Solver as marmote_vi
from solvers.marmote_vigs import Solver as marmote_vigs
from solvers.marmote_pim import Solver as marmote_pim
from solvers.marmote_pimgs import Solver as marmote_pimgs
from utils.generic_model import GenericModel
import pandas as pd

RES_PATH = "best_solver.csv"


def get_best_solver(model: GenericModel, discount: float, precision: float) -> str:
    solvers = {
        "marmote_vi": marmote_vi,
        "marmote_vigs": marmote_vigs,
        "marmote_pim": marmote_pim,
        "marmote_pimgs": marmote_pimgs,
    }

    best_solver_name = None
    best_runtime = float("-inf")

    for name, solver in solvers.items():
        solver_instance = solver(model, discount, precision)
        solver_instance.run()
        runtime = solver_instance.runtime

        if runtime > best_runtime:
            best_runtime = runtime
            best_solver_name = name

    return best_solver_name


def experience(model_before_build):
    df = pd.read_csv(RES_PATH)
    states = [100, 1000, 10000, 20000]
    discounts = [0.5, 0.8, 0.9, 0.99, 0.999]
    final_precisions = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    for state in states:
        model = model_before_build(state, 10)
        model.create_model()
        for discount in discounts:
            for precision in final_precisions:
                best_solver = get_best_solver(model, discount, precision)
                df.loc[len(df)] = (
                    [model.name]
                    + model.get_model_main_parameters()
                    + [precision, discount, best_solver]
                )
            df.to_csv(RES_PATH, index=False)

    df.to_csv(RES_PATH, index=False)



from models.inventory import Model as model_before_build

experience(model_before_build)
