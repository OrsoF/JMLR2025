from solvers import aggregated_qvi
from utils.data_management import solve

model, solver = solve("taxi", "aggregated_vi", 500, 10, 0.99, 1e-2)
print("Etats dans le modèle : {}".format(model.state_dim))
print("Régions : {}".format(solver.partition._number_of_regions()))
print()
for region_index in range(solver.partition._number_of_regions()):
    print("Region {}".format(region_index))
    for state_index in solver.partition.states_in_region[region_index]:
        state_description = model.state_list[state_index]
        print("State :")
        print(
            "Taxi position (x, y) : ({}, {})".format(
                state_description[0], state_description[1]
            )
        )
        print("Passenger position : {}".format(state_description[2]))
        print("Destination : {}".format(state_description[3]))
        print()
    print()
    print()
