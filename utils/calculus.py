import numpy as np
from utils.generic_model import GenericModel
from typing import Tuple, List, Optional
from scipy.sparse import lil_matrix
from utils.exact_value_function import get_exact_value
from utils.partition_generic import GenericPartition

rng = np.random.default_rng(0)


def norminf(value: np.ndarray) -> float:
    """
    Infinite norm of a vector.
    """
    return np.absolute(value).max()


def optimal_bellman_operator(
    model: GenericModel, value: np.ndarray, discount: float
) -> np.ndarray:
    """
    Returns T*(value)
    """
    Q = np.empty((model.action_dim, model.state_dim))
    for aa in range(model.action_dim):
        Q[aa] = model.reward_matrix[:, aa] + discount * model.transition_matrix[aa].dot(
            value
        )
    return Q.max(axis=0)


def q_optimal_bellman_operator(
    model: GenericModel, q_value: np.ndarray, discount: float
) -> np.ndarray:
    """
    Returns T*(q_value)
    """
    value = q_value.max(axis=1)
    q_value_new = np.zeros((model.state_dim, model.action_dim))
    for aa in range(model.action_dim):
        q_value_new[:, aa] = model.reward_matrix[
            :, aa
        ] + discount * model.transition_matrix[aa].dot(value)
    return q_value_new


def bellman_operator(model: GenericModel, value: np.ndarray, discount: float):
    """
    Returns R + discount * T @ V
    """
    q_value = np.zeros((model.state_dim, model.action_dim))
    for aa in range(model.action_dim):
        q_value[:, aa] = model.reward_matrix[
            :, aa
        ] + discount * model.transition_matrix[aa].dot(value)
    return q_value


def bellman_policy_operator(
    value: np.ndarray,
    discount: float,
    transition_policy: np.ndarray,
    reward_policy: np.ndarray,
) -> np.ndarray:
    """
    Returns R^pi + gamma . T^pi @ value
    """
    return reward_policy + discount * transition_policy.dot(value)


def iterative_policy_evaluation(
    transition_policy: np.ndarray,
    reward_policy: np.ndarray,
    discount: float,
    variation_tol: float,
    initial_value: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply Bellman^pi to initial_value until |initial_value - Bellman^pi initial_value| < tolerance
    """
    if initial_value is None:
        value = np.zeros((transition_policy.shape[0]))
    else:
        value = initial_value

    while True:
        next_value = bellman_policy_operator(
            value, discount, transition_policy, reward_policy
        )
        distance = norminf(next_value - value)
        value = next_value

        if distance < variation_tol:
            break

    return value


def full_iterative_policy_evaluation(
    model: GenericModel,
    discount: float,
    policy: np.ndarray,
    variation_tol: float = 1e-2,
):
    transition_policy, reward_policy = compute_transition_reward_policy(model, policy)
    return iterative_policy_evaluation(
        transition_policy, reward_policy, discount, variation_tol
    )


def compute_transition_reward_policy(model: GenericModel, policy: np.ndarray) -> Tuple:
    """
    Given T, R, policy, returns T^pi, R^pi.
    """
    # model._model_to_sparse()
    transition_policy = lil_matrix((model.state_dim, model.state_dim))
    reward_policy = np.zeros(model.state_dim)
    for aa in range(model.action_dim):
        ind = (policy == aa).nonzero()[0]
        if ind.size > 0:
            for ss in ind:
                transition_policy[[ss], :] = model.transition_matrix[aa][
                    [ss], :
                ].toarray()
            reward_policy[ind] = model.reward_matrix[ind, aa]
    transition_policy = transition_policy.tocsr()
    # model._model_to_numpy()
    return transition_policy, reward_policy


def compute_transition_reward_policy_numpy(model: GenericModel, policy: np.ndarray):
    transition_policy = np.zeros((model.state_dim, model.state_dim))
    reward_policy = np.zeros(model.state_dim)
    for aa in range(model.action_dim):
        ind: np.ndarray = (policy == aa).nonzero()[0]
        if ind.size > 0:
            transition_policy[ind, :] = model.transition_matrix[aa][ind, :]
            reward_policy[ind] = model.reward_matrix[ind, aa]
    return transition_policy, reward_policy


def value_span_on_regions(
    full_value: np.ndarray, states_in_regions: List[List[int]]
) -> List[float]:
    """
    Returns [max full_value_k - min full_value_k for k in range(K)]
    """
    return [np.ptp(full_value[region]) for region in states_in_regions]


def is_policy_optimal(
    model: GenericModel, policy: np.ndarray, discount: float, epsilon: float = 1e-2
) -> bool:
    """
    Check if the policy is optimal by comparing the value of the policy with the value of the optimal policy.
    """
    transition_policy, reward_policy = compute_transition_reward_policy(model, policy)
    value_policy = iterative_policy_evaluation(
        transition_policy, reward_policy, discount, epsilon
    )
    q_pi = bellman_operator(model, value_policy, discount)
    new_policy = q_pi.argmax(axis=1)
    return np.array_equal(new_policy, policy)


def q_value_span_on_regions(
    full_q_value: np.ndarray, states_in_regions: List[List[int]]
) -> np.ndarray:
    """
    Returns an array of shape (region_number, action_dim)
    """
    region_number, action_dim = len(states_in_regions), full_q_value.shape[1]
    spans_of_q = np.zeros((region_number, action_dim))
    for region_index, region in enumerate(states_in_regions):
        spans_of_q[region_index, :] = full_q_value[region].max(axis=0) - full_q_value[
            region
        ].min(axis=0)
    return spans_of_q


def get_value_policy_value(
    model: GenericModel,
    discount: float,
    value: np.ndarray,
    precision: float = 1e-3,
) -> np.ndarray:
    """
    For any V, returns V^{pi_V}
    """
    model.test_model()
    assert (
        value.shape[0] == model.state_dim
    ), "Shape of the value should be ({},) instead of {}.".format(
        model.state_dim, value.shape
    )
    policy = bellman_operator(model, value, discount).argmax(axis=1)
    transition_policy, reward_policy = compute_transition_reward_policy(model, policy)
    value_policy = iterative_policy_evaluation(
        transition_policy, reward_policy, discount, precision, value
    )
    return value_policy


def apply_obo_until_var_small(
    model: GenericModel,
    discount: float,
    variation_tol: float,
    initial_value: np.ndarray,
) -> Tuple[np.ndarray, float]:
    value = initial_value

    while True:
        new_value = optimal_bellman_operator(model, value, discount)
        variation = norminf(new_value - value)
        value = new_value
        if variation < variation_tol:
            break

    return value, variation


def generate_sars(model: GenericModel, state: int, action: int) -> tuple[int, float]:
    transition_probabilities = model.transition_matrix[action, state]
    next_state = rng.choice(range(model.state_dim), p=transition_probabilities)
    reward = model.reward_matrix[state, action]
    return next_state, reward


def generate_trajectory(
    model: GenericModel, traj_length: int, best_action: float, q_value: np.ndarray
) -> List:
    """Generate a trajectory of (state, action, reward, next_state) tuples."""
    assert isinstance(model.transition_matrix, np.ndarray), "Numpy model required."
    trajectory = []
    state = rng.integers(model.state_dim)
    for index in range(traj_length):
        action = (
            rng.integers(model.action_dim)
            if rng.random() < best_action
            else np.argmax(q_value[state])
        )
        next_state, reward = generate_sars(model, state, action)
        trajectory.append((state, action, reward, next_state))
        # if next_state == state:
        #     next_state = rng.integers(model.state_dim)
        state = next_state
    return trajectory


def convert_trajectory_to_region(trajectory: list, partition: GenericPartition) -> List:
    """
    Convert a trajectory of (state, action, reward, next_state) tuples to (region, action, reward, next_region).
    """
    converted_trajectory = []
    for state, action, reward, next_state in trajectory:
        region = partition.get_region_index(state)
        next_region = partition.get_region_index(next_state)
        converted_trajectory.append((region, action, reward, next_region))
    return converted_trajectory


def generate_trajectory_partition(
    model: GenericModel,
    partition: GenericPartition,
    traj_length: int,
    best_action: float,
    q_value: np.ndarray,
):
    trajectory = []
    state = rng.integers(model.state_dim)
    state_region = partition.get_region_index(state)
    for index in range(traj_length):
        action = (
            rng.integers(model.action_dim)
            if rng.random() < best_action
            else np.argmax(q_value[state])
        )
        next_state, reward = generate_sars(model, state, action)
        next_state_region = partition.get_region_index(next_state)

        trajectory.append((state_region, action, reward, next_state_region))
        if next_state == state:
            next_state = rng.integers(model.state_dim)
        state = next_state
        state_region = next_state_region
    return trajectory


def generate_sars_random_trajectory_epsilon_greedy(
    model: GenericModel, start_state: int, traj_length: int
) -> List[Tuple[int, float]]:
    """Generate a trajectory of (state, action, reward, next_state) tuples."""
    assert isinstance(model.transition_matrix, np.ndarray), "Numpy model required."
    trajectory = []
    state = start_state
    for _ in range(traj_length):
        action = rng.integers(model.action_dim)  # Random action
        next_state, reward = generate_sars(model, state, action)
        trajectory.append((state, action, reward, next_state))
        state = next_state
    return trajectory


def optimal_bellman_residual(model: GenericModel, value: np.ndarray, discount: float):
    """Returns ||V - T^* V||_inf."""
    return norminf(optimal_bellman_operator(model, value, discount) - value)


def bellman_no_max(
    model: GenericModel, value: np.ndarray, discount: float
) -> np.ndarray:
    """
    For a given value V, returns (R + gamma * T @ V)
    """
    q_value = np.empty((model.state_dim, model.action_dim))
    for aa in range(model.action_dim):
        q_value[:, aa] = (
            model.reward_matrix[:, aa] + discount * model.transition_matrix[aa] @ value
        )
    return q_value


def get_optimal_policy(model: GenericModel, discount: float) -> np.ndarray:
    value = get_exact_value(model, discount)
    q_value = bellman_no_max(model, value, discount)
    return q_value.argmax(axis=1)


def policy_bellman_value_to_value(
    env: GenericModel,
    value: np.ndarray,
    policy: np.ndarray,
    discount: float,
) -> np.ndarray:
    """
    Returns T^pi(value) for given pi, value.
    """
    q_value = np.array(
        [
            env.reward_matrix[:, aa] + discount * (env.transition_matrix[aa].dot(value))
            for aa in range(env.action_dim)
        ]
    )

    return q_value.max(axis=0)


def policy_evaluation(
    env: GenericModel, policy: np.ndarray, discount: float, epsi: float = 1e-2
):
    """
    Evaluate the policy using the Bellman operator until convergence.
    """
    assert policy.shape == (
        env.state_dim,
        env.action_dim,
    ), "Policy shape is {} instead of {}".format(
        policy.shape, (env.state_dim, env.action_dim)
    )
    value = np.zeros((env.state_dim,))

    while True:
        value_old = value.copy()
        value = policy_bellman_value_to_value(env, value, policy, discount)

        if np.linalg.norm(value - value_old, ord=np.inf) < epsi:
            return value


def q_value_block_update(
    q_value, current_region, action, lr, reward, next_region, discount
):
    q_value[current_region, action] += lr * (
        reward + discount * q_value[next_region].max() - q_value[current_region, action]
    )


def q_learning_update(
    model: GenericModel,
    q_value: np.ndarray,
    lr: float,
    state: int,
    best_action: float,
    discount: float,
) -> Tuple[np.ndarray, int]:
    action = (
        np.argmax(q_value[state])
        if rng.random() < best_action
        else rng.integers(model.action_dim)
    )
    next_state, reward = generate_sars(model, state, action)
    q_value[state, action] += lr * (
        reward + discount * q_value[next_state].max() - q_value[state, action]
    )
    return q_value, next_state


def q_value_block_update(
    q_value: np.ndarray,
    partition: GenericPartition,
    current_state: int,
    action: int,
    lr: float,
    reward: float,
    next_state: int,
    discount: float,
) -> np.ndarray:
    """
    Update the q_value by block using the partition. To be improved using np array properties.
    """
    current_region_index = partition.get_region_index(current_state)

    delta = (
        reward + discount * q_value[next_state].max() - q_value[current_state, action]
    )
    q_value[partition.states_in_region[current_region_index], action] += lr * delta
    return q_value


def ql_block_update_step(
    model: GenericModel,
    partition: GenericPartition,
    q_value: np.ndarray,
    lr: float,
    current_state: int,
    epsilon: float,
    discount: float,
):

    action = (
        rng.integers(model.action_dim)
        if rng.random() < epsilon
        else np.argmax(q_value[current_state])
    )
    next_state, reward = generate_sars(model, current_state, action)
    current_region = partition.get_region_index(current_state)
    delta = (
        reward + discount * q_value[next_state].max() - q_value[current_state, action]
    )
    q_value[partition.states_in_region[current_region], action] += lr * delta

    return q_value, next_state


def ql_block_update_along_trajectory(
    model: GenericModel,
    partition: GenericPartition,
    q_value: np.ndarray,
    lr: float,
    epsilon: float,
    discount: float,
    start_state: int,
    traj_length: int,
):
    state = start_state
    for _ in range(traj_length):

        action = (
            rng.integers(model.action_dim)
            if rng.random() < epsilon
            else np.argmax(q_value[state])
        )
        next_state, reward = generate_sars(model, state, action)
        current_region = partition.get_region_index(state)
        delta = reward + discount * q_value[next_state].max() - q_value[state, action]
        q_value[partition.states_in_region[current_region], action] += lr * delta

        state = next_state

    return q_value


def bellman_residual_policy(
    model: GenericModel, policy: np.ndarray, value: np.ndarray, discount: float
) -> float:
    """Compute the Bellman residual for a given policy and value function."""
    transition_policy, reward_policy = compute_transition_reward_policy(model, policy)
    bellman_value = reward_policy + discount * transition_policy.dot(value)
    return norminf(bellman_value - value)


def is_policy_optimal(
    model: GenericModel, policy: np.ndarray, discount: float, epsilon: float = 1e-2
) -> bool:
    """ "Check if the policy is optimal."""
    policy_value = full_iterative_policy_evaluation(
        model, discount, policy, variation_tol=epsilon
    )
    q_value = bellman_no_max(model, policy_value, discount)
    new_policy = q_value.argmax(axis=1)

    return np.all(new_policy == policy)
