from panel import state
from utils.data_management import solve

models = [
    ("rooms_total", 250000),
    ("mountain_total", 100000),
    ("taxi_total", 250000),
]

solvers = [
    "aggregated_vi_total",
    "aggregated_qvi_total",
    "personal_vi",
    "aggregated_pim_total",
    "personal_pim",
]

action_dim = 20  # Default action dimension, this parameter changes depending the model.
precision = 1e-3
repeat = 5
discount = 1.0 # Total reward setting

for model_name, state_dim in models:
    print(
        f"Model {model_name}, State Dimension: {state_dim}, Precision: {precision}, Repeat: {repeat}"
    )
    for solver_name in solvers:
        if state_dim > 15000 and 'pim' in solver_name:
            continue
        solve(
            model_name, solver_name, state_dim, action_dim, discount, precision, repeat
        )
    print()
