from utils.data_management import solve

model, solver = solve('tandem', 'mdptoolbox_vi', 500, 20, 0.9, 1e-2)
print(model.name)
model.test_model(show_progression_bar=True)

print(solver.distance_to_optimal())
solver.plot_value_function()