from models.rooms import Model
import numpy as np

model = Model(10000, 10)
model.create_model()

from utils.calculus import is_policy_optimal
from utils.exact_value_function import get_optimal_policy

discounts = np.round(1 - 10**(-np.linspace(0.5, 3, 5)), 3)

for discount1 in discounts:
    optimal_policy = get_optimal_policy(model, discount1)
    result = is_policy_optimal(model, optimal_policy, max(discounts))
    print(f"Is the policy optimal for discount factor {discount1} with respect to {max(discounts)}? {result}")
