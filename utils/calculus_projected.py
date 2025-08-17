from calendar import c
from gurobipy import max_
import numpy as np
from utils.generic_model import GenericModel, NUMPY, SPARSE
from typing import List, Tuple, Optional
from utils.calculus import norminf
from scipy.sparse import csr_matrix, csr_array

seed = 0
rnd_gen = np.random.default_rng(seed)


def projected_optimal_bellman_operator(
    model: GenericModel,
    discount: float,
    contracted_value: np.ndarray,
    aggregated_transition: List[np.ndarray],
    aggregated_reward: np.ndarray,
    weights,
) -> np.ndarray:
    """
    Returns w @ max_a(agg_R + gamma * agg_T @ agg_V)
    """
    contracted_value = contracted_value.squeeze()
    contracted_q_value = np.empty((model.state_dim, model.action_dim))

    for aa in range(model.action_dim):
        contracted_q_value[:, aa] = aggregated_reward[
            :, aa
        ] + discount * aggregated_transition[aa].dot(contracted_value)

    return weights.dot(contracted_q_value.max(axis=1))


def apply_pobo_until_var_small(
    model: GenericModel,
    discount: float,
    aggregated_transition: List,
    aggregated_reward: np.ndarray,
    weights,
    variation_tol: float,
    initial_contracted_value: Optional[np.ndarray] = None,
    norm_weights: Optional[np.ndarray] = None,
    max_steps=np.inf,
):

    contracted_value = (
        initial_contracted_value
        if initial_contracted_value is not None
        else np.zeros((aggregated_transition[0].shape[0]))
    )

    while True:
        new_contracted_value = projected_optimal_bellman_operator(
            model,
            discount,
            contracted_value,
            aggregated_transition,
            aggregated_reward,
            weights,
        )
        variation = norminf(new_contracted_value - contracted_value)
        contracted_value = new_contracted_value
        if variation < variation_tol:
            break

    return contracted_value, variation


def apply_pobo_until_var_small_norm(
    model: GenericModel,
    discount: float,
    aggregated_transition: List,
    aggregated_reward: np.ndarray,
    weights,
    variation_tol: float,
    initial_contracted_value: Optional[np.ndarray] = None,
    norm_weights: Optional[np.ndarray] = None,
    max_steps=np.inf,
):
    steps_done = 0

    if initial_contracted_value is None:
        contracted_value = np.zeros((aggregated_transition[0].shape[1]))
    else:
        contracted_value = initial_contracted_value

    while True:
        steps_done += 1

        new_contracted_value = projected_optimal_bellman_operator(
            model,
            discount,
            contracted_value,
            aggregated_transition,
            aggregated_reward,
            weights,
        )
        if norm_weights is None:
            variation = norminf(new_contracted_value - contracted_value)
        else:
            variation = norminf(
                norm_weights.dot((new_contracted_value - contracted_value))
            )
        contracted_value = new_contracted_value
        if variation < variation_tol or steps_done > max_steps:
            break

    return contracted_value, variation


def projected_optimal_bellman_operator_residual(
    model: GenericModel,
    discount: float,
    contracted_value: np.ndarray,
    aggregated_transition: List[np.ndarray],
    aggregated_reward: np.ndarray,
    weights,
):
    return norminf(
        contracted_value
        - projected_optimal_bellman_operator(
            model,
            discount,
            contracted_value,
            aggregated_transition,
            aggregated_reward,
            weights,
        )
    )


def projected_optimal_q_bellman_operator(
    model: GenericModel,
    discount: float,
    contracted_q_value: np.ndarray,
    aggregated_transition: List[np.ndarray],
    aggregated_reward: np.ndarray,
) -> np.ndarray:
    """
    Returns w @ max_a(agg_R + gamma * agg_T @ agg_V)
    """

    region_number = contracted_q_value.shape[0]

    contracted_value = contracted_q_value.max(axis=1)

    contracted_q_value = np.empty((region_number, model.action_dim))
    for aa in range(model.action_dim):
        contracted_q_value[:, aa] = aggregated_reward[
            :, aa
        ] + discount * aggregated_transition[aa].dot(contracted_value)

    return contracted_q_value


def apply_poqbo_until_var_small(
    model: GenericModel,
    discount: float,
    aggregated_transition: List[np.ndarray],
    aggregated_reward: np.ndarray,
    variation_tol: float,
    init_agg_qvalue: Optional[np.ndarray] = None,
    norm_weights: Optional[np.ndarray] = None,
    max_steps=np.inf,
) -> Tuple[np.ndarray, float]:
    """
    Apply Pi T_Q until |V - Pi T_Q V| < variation_tol.
    """
    steps_done = 0

    q_contracted_value = (
        np.zeros((aggregated_reward.shape[0]))
        if init_agg_qvalue is None
        else init_agg_qvalue
    )

    while True:
        steps_done += 1

        new_q_contracted_value = projected_optimal_q_bellman_operator(
            model,
            discount,
            q_contracted_value,
            aggregated_transition,
            aggregated_reward,
        )
        if norm_weights is None:
            variation = norminf(new_q_contracted_value - q_contracted_value)
        else:
            variation = norminf(
                norm_weights.dot(new_q_contracted_value - q_contracted_value)
            )
        q_contracted_value = new_q_contracted_value
        if variation < variation_tol or steps_done > max_steps:
            break

    return q_contracted_value, variation


def projected_policy_bellman_operator(
    discount: float,
    contracted_value: np.ndarray,
    aggregated_transition_policy: np.ndarray,
    aggregated_reward_policy: np.ndarray,
) -> np.ndarray:
    """
    Returns (w R^pi phi) + discount * (w T^pi phi) V
    """
    return aggregated_reward_policy + discount * aggregated_transition_policy.dot(
        contracted_value
    )


def apply_ppbo_until_var_small(
    discount: float,
    agg_trans_pi: np.ndarray,
    agg_rew_pi: np.ndarray,
    variation_tol: float,
    init_agg_value: np.ndarray,
    max_steps: int,
) -> Tuple[np.ndarray, float]:
    """
    Applies Pi T^pi until ||V - Pi T^pi V|| <= variation_tol.
    """
    contracted_value = init_agg_value
    steps_done = 0

    while True:
        steps_done += 1
        new_contracted_value = projected_policy_bellman_operator(
            discount,
            contracted_value,
            agg_trans_pi,
            agg_rew_pi,
        )
        variation = norminf(new_contracted_value - contracted_value)
        contracted_value = new_contracted_value
        if variation < variation_tol or steps_done > max_steps:
            break

    return contracted_value, variation


def create_random_partition(model: GenericModel, region_number: int) -> list:
    """
    Create a random partition
    for a given number of regions.
    """
    states_in_region = [[k] for k in range(region_number)]
    for state in range(region_number, model.state_dim):
        k = rnd_gen.integers(region_number)
        states_in_region[k].append(state)
    return states_in_region


def check_dimensions(
    model: GenericModel,
    contracted_value: np.ndarray,
    aggregated_transition: list,
    aggregated_reward: np.ndarray,
):
    region_number = contracted_value.shape[0]
    assert len(aggregated_transition) == model.action_dim
    assert aggregated_transition[0].shape == (
        model.state_dim,
        region_number,
    ), "Wrong transition shape : {} != {}".format(
        aggregated_transition[0].shape, (model.state_dim, region_number)
    )
    assert aggregated_reward.shape == (model.state_dim, model.action_dim)
