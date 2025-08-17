import numpy as np
from typing import List, Tuple
from utils.generic_model import GenericModel
from utils.calculus_partition import (
    intersection_partitions,
)
from utils.partition_generic import GenericPartition


class PiPartition(GenericPartition):
    """
    We assume a partition is described by
    """

    def __init__(self, model: GenericModel, discount: float, mode: str) -> None:
        self.model = model
        self.discount = discount
        self.mode = mode

        self._initialize_trivial_aggregation()

    def projected_bellman_policy_operator(self, contracted_value: np.ndarray):
        """
        Get (Pi T)(contracted_value)
        """
        # self._compute_aggregate_transition_reward_policy()
        self._build_weights()

        contracted_value = (
            self.aggregate_reward_policy
            + self.discount * self.aggregate_transition_policy @ contracted_value
        )
        return contracted_value

    def divide_regions_along_tv(
        self,
        tv_vector: np.ndarray,
        epsilon: float,
        contracted_value_to_update: np.ndarray,
        check_dimension: bool = True,
    ):
        if check_dimension:
            assert len(tv_vector.shape) == 1, "{}".format(tv_vector.shape)
        new_partition_dictionary = self._build_new_partition_dictionary(
            tv_vector, epsilon
        )
        new_contracted_value = self._update_partition_and_value(
            new_partition_dictionary, contracted_value_to_update
        )
        del self.aggregate_transition_policy
        del self.aggregate_reward_policy
        del self.full_phi
        del self.weights

        return new_contracted_value

    def _build_new_partition_dictionary(self, tv_vector: np.ndarray, epsilon: float):
        new_partition_dictionary: dict[int, list]
        new_partition_dictionary = {
            region_index: [region]
            for region_index, region in enumerate(self.states_in_region)
        }

        for region_index, region in enumerate(self.states_in_region):
            local_value = tv_vector[region]
            if np.ptp(local_value) > epsilon:
                new_partition = self._partition_along_value(
                    local_value, region, epsilon
                )
                if len(new_partition_dictionary[region_index]):
                    new_partition_dictionary[region_index] = intersection_partitions(
                        new_partition_dictionary[region_index], list(new_partition)
                    )
                else:
                    new_partition_dictionary[region_index] = new_partition

        return new_partition_dictionary

    def _partition_along_value(
        self,
        local_value: np.ndarray,
        region: List[int],
        epsilon: float,
        check_dimension: bool = True,
    ):
        if check_dimension:
            assert (
                len(local_value.shape) == 1
            ), "local_value should be of dimension 1. {}".format(local_value.shape)

        min_local_value = local_value.min()
        partition: dict[int, list[int]] = {}
        for index, val in enumerate(local_value):
            region_state = region[index]
            interval_index = int((val - min_local_value) / epsilon)

            if interval_index in partition:
                partition[interval_index].append(region_state)
            else:
                partition[interval_index] = [region_state]
        return partition.values()

    def _update_partition_and_value(
        self, new_partition_dictionary: dict, contracted_value_to_update: np.ndarray
    ) -> np.ndarray:
        for region_index, partition in new_partition_dictionary.items():
            self.states_in_region[region_index] = partition[0]
            self.states_in_region.extend(partition[1:])

            value_to_add = np.tile(
                contracted_value_to_update[region_index], (len(partition) - 1)
            )

            contracted_value_to_update = np.concatenate(
                (contracted_value_to_update, value_to_add), axis=0
            )

        return contracted_value_to_update

    def update_transition_reward_policy(
        self, transition_policy: np.ndarray, reward_policy: np.ndarray
    ):
        self.transition_policy = transition_policy
        self.reward_policy = reward_policy
        self._compute_aggregate_transition_reward_policy()

    def _compute_aggregate_transition_reward_policy(
        self, recompute=False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (w @ T^pi @ phi), (w @ R^pi)
        """
        if (
            not hasattr(self, "aggregate_transition_policy")
            and not hasattr(self, "aggregate_reward_policy")
        ) or recompute:
            self._compute_weights_phi()
            partial_phi = self._partial_phi()

            self.aggregate_transition_policy = (
                self.weights @ self.transition_policy @ partial_phi
            )

            self.aggregate_reward_policy = self.weights @ self.reward_policy

            return self.aggregate_transition_policy, self.aggregate_reward_policy

        elif (
            self.aggregate_transition_policy is not None
            and self.aggregate_reward_policy is not None
        ):
            return self.aggregate_transition_policy, self.aggregate_reward_policy

        else:
            assert False, "Impossible case."
