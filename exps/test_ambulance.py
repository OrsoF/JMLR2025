from utils.data_management import solve

# model, solver = solve('ambulance', 'personal_vi', 500, 10, 0.1, 1e-2)
# # print(solver.distance_to_optimal())
# # solver.plot_value_function()
# model.test_model()

from models.ambulance import Model

model = Model(100, 10)
model.create_model()
model.test_model()