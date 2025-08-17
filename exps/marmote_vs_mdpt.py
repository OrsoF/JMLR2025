import numpy as np
from scipy.sparse import random
from marmote.core import MarmoteInterval, SparseMatrix, FullMatrix
from marmote.mdp import DiscountedMDP, FeedbackSolutionMDP, TotalRewardMDP
from time import time
import numpy as np
from mdptoolbox.mdp import PolicyIterationModified
from tqdm import trange, tqdm

# Création du modèle

state_dim = 1000
action_dim = 10
discount = 0.999
sparsity_transition = 0.85
reward_std = 1.0

epsilon = 1e-1
max_policy_update = int(10**8)
delta = 1e-1
max_step_policy_evaluation = int(10**5)

random_generator = np.random.default_rng(seed=0)

reward_matrix = reward_std * random_generator.standard_normal(
    size=(state_dim, action_dim)
)
transition_matrix = [
    random(
        state_dim,
        state_dim,
        density=sparsity_transition,
        format="lil",
        data_rvs=np.random.rand,
    )
    for _ in range(action_dim)
]

for aa in range(action_dim):
    for ss in range(state_dim):
        transition_matrix[aa][ss, 0] += 1e-6
        transition_matrix[aa][ss] *= 1 / transition_matrix[aa][ss].sum()
        transition_matrix[aa][ss, 0] += 1 - transition_matrix[aa][ss].sum()

for aa in range(action_dim):
    for ss in range(state_dim):
        transition_matrix[aa][ss, 0] += 1 - transition_matrix[aa][ss].sum()

transition_matrix = [transition.tocsr() for transition in transition_matrix]


# Résolution MDPToolbox

pim = PolicyIterationModified(
    transition_matrix,
    reward_matrix,
    discount,
    epsilon,
    max_policy_update,
    skip_check=True,
)
pim.run()
print("MDPToolbox runtime {}".format(pim.time))


# Résolution Marmote


def build_marmote_reward_matrix(
    state_dim: int, action_dim: int, reward_matrix: np.ndarray
) -> FullMatrix:
    marmote_reward_matrix = FullMatrix(state_dim, action_dim)
    for a in range(action_dim):
        for s in range(state_dim):
            marmote_reward_matrix.setEntry(s, a, float(reward_matrix[s, a]))
    return marmote_reward_matrix


def build_marmote_transition_list(
    state_dim: int, action_dim: int, transition_matrix: list[np.ndarray]
) -> list:
    marmote_transitions_list = list()
    for aa in range(action_dim):
        P = SparseMatrix(state_dim)

        row_indices, col_indices = transition_matrix[aa].nonzero()
        for i in range(len(row_indices)):
            ss1 = row_indices[i]
            ss2 = col_indices[i]
            val = transition_matrix[aa][ss1, ss2]
            P.addToEntry(int(ss1), int(ss2), val)

        marmote_transitions_list.append(P)
        P = None
    return marmote_transitions_list


state_space = MarmoteInterval(0, int(state_dim - 1))
action_space = MarmoteInterval(0, int(action_dim - 1))

reward_matrix_marmote = build_marmote_reward_matrix(
    state_dim, action_dim, reward_matrix
)

transitions_list_marmote = build_marmote_transition_list(
    state_dim, action_dim, transition_matrix
)

mdp = DiscountedMDP(
    "max",
    state_space,
    action_space,
    transitions_list_marmote,
    reward_matrix_marmote,
    discount,
)

mdp.changeVerbosity(True)

start_time = time()

opt: FeedbackSolutionMDP = mdp.PolicyIterationModified(
    epsilon,
    max_policy_update,
    delta,
    max_step_policy_evaluation,
)

runtime = time() - start_time

value = np.array([opt.getValueIndex(ss) for ss in range(state_dim)])
policy = None

total_time = time() - start_time

print("Marmote runtime {}".format(total_time))
