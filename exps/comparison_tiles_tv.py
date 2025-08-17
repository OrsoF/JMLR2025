from utils.data_management import solve

model_name = "oil"
discount = 0.9
precision = 1e-2
state_dim = 10000
action_dim = 10
repeat = 1

model, solver = solve(
    model_name=model_name,
    solver_name="aggregated_pim",
    state_dim=state_dim,
    action_dim=action_dim,
    discount=discount,
    precision=precision,
    repeat=repeat,
)
print("TV :    {}".format(solver.runtime))

model, solver = solve(
    model_name=model_name,
    solver_name="aggregated_pim_tiles",
    state_dim=state_dim,
    action_dim=action_dim,
    discount=discount,
    precision=precision,
    repeat=repeat,
)

print("Tiles : {}".format(solver.runtime))

model, solver = solve(
    model_name=model_name,
    solver_name="personal_pim",
    state_dim=state_dim,
    action_dim=10,
    discount=discount,
    precision=precision,
    repeat=repeat,
)
print("PIM : {}".format(solver.runtime))
