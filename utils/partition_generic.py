from utils.generic_model import GenericModel
from utils.calculus_partition import (
    get_full_phi_from_states_in_regions,
    get_weight_matrix_from_states_in_regions,
    get_weights_from_partial_phi,
)
import numpy as np
from math import floor
from typing import Tuple

seed = 0
rnd_gen = np.random.default_rng(seed=seed)


class GenericPartition:
    def __init__(self, model: GenericModel, discount: float):
        self.model = model
        self.discount = discount

        self.states_in_region: list = [list(range(self.model.state_dim))]
        self.aggregate_transition_matrix: list
        self.aggregate_reward_matrix: np.ndarray

    def _number_of_regions(self) -> int:
        """Compute the number of regions in the partition."""
        return len(self.states_in_region)

    def _initialize_trivial_aggregation(self):
        """Initialize the partition with one region : the whole state space."""
        self.states_in_region = [list(range(self.model.state_dim))]

    def _partial_phi(self):
        """Returns the phi matrix. Require phi to be built."""
        return self.full_phi[:, : len(self.states_in_region)]

    def _compute_weights_phi(self, force: bool = False):
        """Compute omega and phi matrices using self.states_in_regions."""
        if hasattr(self, "full_phi") and hasattr(self, "weights") and not force:
            return
        self.full_phi = get_full_phi_from_states_in_regions(
            self.model, self.states_in_region
        )
        self.weights = get_weights_from_partial_phi(self._partial_phi(), False)

    def _compute_phi(self):
        """Compute only phi matrix using self.states_in_regions."""
        self.full_phi = get_full_phi_from_states_in_regions(
            self.model, self.states_in_region
        )

    def _find_region(self, state_index: int) -> int:
        """For a given state index, find the current region it is in. Require phi to be calculated."""
        return np.argmax(self._partial_phi()[state_index])

    def divide_region_update_contracted_value(
        self,
        region_index: int,
        new_partition: list,
        contracted_value: list,
        check_arg: bool = False,
    ):
        """
        Given a region region_index, divide it into the new_partition.
        Update contracted_value at the same time.
        """
        if check_arg:
            assert len(new_partition) > 0
            for region in new_partition:
                for state in region:
                    assert state in self.states_in_region[region_index]
            assert len(self.states_in_region[region_index]) == sum(
                len(region) for region in new_partition
            )

        if len(new_partition) == 1:
            return contracted_value
        else:
            self.states_in_region[region_index] = new_partition[0]
            self.states_in_region = self.states_in_region + new_partition[1:]

            contracted_value = contracted_value + [contracted_value[region_index]] * (
                len(new_partition) - 1
            )
            return contracted_value

    def divide_region_along_value(
        self, region: list, value: np.ndarray, epsilon: float
    ) -> list:
        """
        Take one region and divide it along value with step epsilon.
        """
        new_partition_dict = {}
        for state_index, state in enumerate(region):
            index = floor(value[state_index] / epsilon)
            if index in new_partition_dict:
                new_partition_dict[index].append(state)
            else:
                new_partition_dict[index] = [state]
        new_partition = [region for region in new_partition_dict.values()]
        return new_partition

    def divide_region_along_value_tiles(
        self, region: list, value: np.ndarray, n_tiles: int
    ) -> list:
        """Divide a single region into n_tiles pieces along local value value."""
        if len(region) == 1:
            return [region]
        elif len(region) == 2:
            return [[region[0]], [region[1]]]
        else:
            region = [(state, value[i]) for i, state in enumerate(region)]
            region.sort(key=lambda elem: elem[1])
            region = [elem[0] for elem in region]
            tile_size = len(region) // n_tiles + 1
            return [
                region[tile_size * i : min(tile_size * (i + 1), len(region))]
                for i in range(n_tiles)
                if len(region[tile_size * i : min(tile_size * (i + 1), len(region))])
            ]

    def divide_all_regions_along_value_update_contracted_value_tiles(
        self,
        value: np.ndarray,
        contracted_value: np.ndarray,
        n_tiles: int,
        check_args: bool = False,
    ) -> np.ndarray:
        if check_args:
            assert (
                len(value) == self.model.state_dim
            ), "Wrong value shape {} != {}".format(len(value), self.model.state_dim)
            assert len(self.states_in_region) == len(
                contracted_value
            ), "Wrong contracted value shape {} != {}".format(
                len(self.states_in_region), len(contracted_value)
            )

        self.reset_attributes()
        contracted_value_list: list = contracted_value.tolist()
        # Compute partitions
        all_new_partition = {}
        for region_index, region in enumerate(self.states_in_region):
            new_partition = self.divide_region_along_value_tiles(
                region, value[region], n_tiles
            )
            all_new_partition[region_index] = new_partition

        # Update state_in_region
        for region_index, new_partition in all_new_partition.items():
            contracted_value_list = self.divide_region_update_contracted_value(
                region_index, new_partition, contracted_value_list
            )
        contracted_value = np.array(contracted_value_list)
        return contracted_value

    def divide_all_regions_along_value_update_contracted_value(
        self,
        value: np.ndarray,
        epsilon: float,
        contracted_value: np.ndarray,
        check_args: bool = False,
    ) -> np.ndarray:
        if check_args:
            assert (
                len(value) == self.model.state_dim
            ), "Wrong value shape {} != {}".format(len(value), self.model.state_dim)
            assert len(self.states_in_region) == len(
                contracted_value
            ), "Wrong contracted value shape {} != {}".format(
                len(self.states_in_region), len(contracted_value)
            )

        self.reset_attributes()
        contracted_value_list: list = contracted_value.tolist()
        # Compute partitions
        all_new_partition = {}
        for region_index, region in enumerate(self.states_in_region):
            new_partition = self.divide_region_along_value(
                region, value[region], epsilon
            )
            all_new_partition[region_index] = new_partition

        # Update state_in_region
        for region_index, new_partition in all_new_partition.items():
            contracted_value_list = self.divide_region_update_contracted_value(
                region_index, new_partition, contracted_value_list
            )
        contracted_value = np.array(contracted_value_list).squeeze()
        return contracted_value

    def divide_all_regions_along_value_update_contracted_q_value(
        self, value: np.ndarray, epsilon: float, contracted_q_value: np.ndarray
    ) -> np.ndarray:
        self.reset_attributes()
        # Compute partitions
        all_new_partition = {}
        for region_index, region in enumerate(self.states_in_region):
            new_partition = self.divide_region_along_value(
                region, value[region], epsilon
            )
            all_new_partition[region_index] = new_partition

        # Update state_in_region
        for region_index, new_partition in all_new_partition.items():
            contracted_q_value = self.divide_region_update_contracted_q_value(
                region_index, new_partition, contracted_q_value
            )
        return contracted_q_value

    def reset_attributes(self):
        """
        Reset partition attributes.
        """
        try:
            del self.aggregate_transition_matrix
        except AttributeError:
            pass
        try:
            del self.aggregate_reward_matrix
        except AttributeError:
            pass
        try:
            del self.full_phi
        except AttributeError:
            pass
        try:
            del self.weights
        except AttributeError:
            pass

    def divide_region_update_contracted_q_value(
        self,
        region_index: int,
        new_partition: list,
        contracted_q_value: np.ndarray,
        check_arg: bool = False,
    ):
        if check_arg:
            assert len(new_partition) > 0
            for region in new_partition:
                for state in region:
                    assert state in self.states_in_region[region_index]
            assert len(self.states_in_region[region_index]) == sum(
                len(region) for region in new_partition
            )

        if len(new_partition) == 1:
            return contracted_q_value
        else:
            self.states_in_region[region_index] = new_partition[0]
            self.states_in_region = self.states_in_region + new_partition[1:]

            tile_to_add = np.repeat(
                contracted_q_value[region_index].reshape((-1, 1)),
                len(new_partition) - 1,
                axis=1,
            ).T

            contracted_q_value = np.concatenate(
                (
                    contracted_q_value,
                    tile_to_add,
                )
            )
            return contracted_q_value

    def compute_agg_trans_reward_v(self):
        if hasattr(self, "aggregate_transition_matrix"):
            return
        else:
            self.aggregate_transition_matrix = [
                self.model.transition_matrix[aa].dot(self._partial_phi())
                for aa in range(self.model.action_dim)
            ]
            self.aggregate_reward_matrix = self.model.reward_matrix
            return self.aggregate_transition_matrix, self.aggregate_reward_matrix

    def compute_agg_trans_reward_q(self) -> Tuple[list, np.ndarray]:
        if hasattr(self, "aggregate_transition_matrix"):
            return self.aggregate_transition_matrix, self.aggregate_reward_matrix
        else:
            self.aggregate_transition_matrix = [
                self.weights.dot(
                    self.model.transition_matrix[aa].dot(self._partial_phi())
                )
                for aa in range(self.model.action_dim)
            ]
            self.aggregate_reward_matrix = self.weights.dot(self.model.reward_matrix)
            return self.aggregate_transition_matrix, self.aggregate_reward_matrix

    def compute_agg_trans_reward_pi(self, transition_policy, reward_policy):
        if hasattr(self, "aggregate_transition_matrix"):
            return
        else:
            self.aggregate_transition_policy = self.weights.dot(
                transition_policy.dot(self._partial_phi())
            )
            self.aggregate_reward_policy = self.weights.dot(reward_policy)
            return self.aggregate_transition_policy, self.aggregate_reward_policy

    def generate_random_partition(self, number_of_regions: int):
        number_of_regions = int(np.clip(number_of_regions, 1, self.model.state_dim))
        all_states = list(range(number_of_regions, self.model.state_dim))
        self.states_in_region = [[state] for state in range(number_of_regions)]
        while len(all_states):
            choosen_state = rnd_gen.choice(all_states)
            all_states.remove(choosen_state)
            self.states_in_region[rnd_gen.integers(number_of_regions)].append(
                choosen_state
            )

    def move_state(self, state: int, origin_region: int, final_region: int) -> None:
        """Move a state from one region to another."""
        assert state in self.states_in_region[origin_region]
        assert state not in self.states_in_region[final_region]
        self.states_in_region[origin_region].remove(state)
        self.states_in_region[final_region].append(state)

    def build_abstract_model(self):
        trans, reward = self.compute_agg_trans_reward_q()
        return CustomModel(trans, reward, "Abstract_MDP")

    def get_region_index(self, state: int) -> int:
        """For a given state, return the region index where this state is."""
        for region_index, region in enumerate(self.states_in_region):
            if state in region:
                return region_index
        raise ValueError(f"State {state} not found in any region.")