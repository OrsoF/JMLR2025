import numpy as np
from typing import List
from utils.generic_model import GenericModel, NUMPY, SPARSE
from scipy.sparse import lil_matrix

UNKNOWN_MODE = "Unknown mode {}."


def span_by_region(
    value_function: np.ndarray, states_in_region: List[List[int]]
) -> List[float]:
    """
    Return a list :
    [max(value[region])-min(value[region]) for region in regions]
    """
    return [np.ptp(value_function[region]) for region in states_in_region]


def intersection_partitions(partition_1: list, partition_2: list) -> list:
    """
    Given two partitions, returns a thinner partition
    intersecting both.
    """
    return [
        list(set(sublist_1) & set(sublist_2))
        for sublist_1 in partition_1
        for sublist_2 in partition_2
        if set(sublist_1) & set(sublist_2)
    ]


def get_weight_matrix_from_states_in_regions(
    state_dim: int, states_in_regions: List[List[int]], mode: str
):
    """
    Get the w matrix s.t. w @ phi = 1 from states_in_regions.
    """
    region_number = len(states_in_regions)

    if mode == NUMPY:
        weight_matrix = np.zeros((region_number, state_dim))
        for region_index in range(region_number):
            region_weight = 1 / len(states_in_regions[region_index])
            weight_matrix[region_index, states_in_regions[region_index]] = region_weight
    elif mode == SPARSE:
        weight_matrix = lil_matrix((region_number, state_dim))
        for region_index in range(region_number):
            region_weight = 1 / len(states_in_regions[region_index])
            weight_matrix[region_index, states_in_regions[region_index]] = region_weight
        weight_matrix = weight_matrix.tocsr()
    else:
        assert False, UNKNOWN_MODE.format(mode)

    return weight_matrix


def get_weights_from_partial_phi(partial_phi: np.ndarray, check_inputs: bool = False):
    """
    Get the w matrix s.t. w @ phi = 1 from the partial_phi matrix.
    """
    if check_inputs:
        assert np.all(
            partial_phi.sum(axis=0) > 0
        ), "Each region should contain at least one state."
    weights = partial_phi.T
    weights = weights / weights.sum(axis=1)
    return weights


def get_full_phi_from_states_in_regions(model: GenericModel, states_in_region: List):
    full_phi = lil_matrix((model.state_dim, model.state_dim))
    for region_index in range(len(states_in_region)):
        for state_index in states_in_region[region_index]:
            full_phi[state_index, region_index] = 1.0
    full_phi = full_phi.tocsr()
    return full_phi
