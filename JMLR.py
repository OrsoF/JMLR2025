from utils.data_management import solve

models = [
    ("rooms", 250000),
    ("mountain", 100000),
    ("taxi", 250000),
    ("barto", 250000),
    ("tandem", 50000),
    ("dam", 16000),
    ("inventory", 10000),
    ("impatience", 1000),
    ("garnet", 500),
    ("block", 500),
]

solvers = [
    "aggregated_vi",
    "aggregated_qvi",
    "personal_vi",
    "chen_td",
    "aggregated_pim",
    "personal_pim",
    "bertsekas_pi",
]

discount = 0.9999
action_dim = 20  # Default action dimension, this parameter changes depending the model.
precision = 1e-3
repeat = 5

for model_name, state_dim in models:
    if state_dim > 100000:
        repeat = 1
    print(
        f"Model {model_name}, State Dimension: {state_dim}, Discount: {discount}, Precision: {precision}, Repeat: {repeat}"
    )
    for solver_name in solvers:
        solve(
            model_name, solver_name, state_dim, action_dim, discount, precision, repeat
        )
    print()
