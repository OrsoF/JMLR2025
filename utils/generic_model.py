import os
import time
import numpy as np
from scipy.sparse import csr_array, lil_matrix
from tqdm import trange

from utils.pickle import pickle_load, pickle_save


NUMPY, SPARSE = "numpy", "sparse"

def norminf(x: np.ndarray) -> float:
    """
    Compute the infinity norm of a vector.
    """
    return np.max(np.abs(x))


class GenericModel:
    def __init__(self, state_dim: int, action_dim: int) -> None:
        # Parsing params
        self.name: str
        self.state_dim: int
        self.action_dim: int
        self.transition_matrix: list | np.ndarray
        self.reward_matrix: np.ndarray
        self.params: dict

    def create_model(
        self,
        check_transition: bool = False,
        normalize_reward: bool = False,
    ):
        """
        Function that create the reward and transition matrices.
        """
        # Saving files process
        self.pickle_file_name = "{}.pkl".format(self.name)
        if "saved_models" not in os.listdir(os.getcwd()):
            os.mkdir("saved_models")

        if hasattr(self, "transition_matrix") and hasattr(self, "reward_matrix"):
            return
        elif self.pickle_file_name not in os.listdir("saved_models"):
            start_build_time = time.time()
            self._build_model()
            build_time = np.round(time.time() - start_build_time, 2)
            print("Build time : {}".format(build_time))

            # print("Model built.")
            if normalize_reward:
                self._normalize_reward_matrix()
            if check_transition:
                self.test_model()
                print("Transition is stochastic.")

            model = (
                self.state_dim,
                self.action_dim,
                self.transition_matrix,
                self.reward_matrix,
            )

            # print("Saving model...")
            pickle_save(model, os.path.join("saved_models", self.pickle_file_name))

        else:
            (
                self.state_dim,
                self.action_dim,
                self.transition_matrix,
                self.reward_matrix,
            ) = pickle_load(os.path.join("saved_models", self.pickle_file_name))

    def _build_model(self):
        # Define it in the specific model.
        GENERIC_METHOD_USAGE_MESSAGE = "This 'build_model' method is the generic one, build a new one in the specific model."
        assert False, GENERIC_METHOD_USAGE_MESSAGE

    def lighten_model(self):
        """
        Use it to remove heavy matrices during computations.
        """
        try:
            del self.transition_matrix
            del self.reward_matrix
        except AttributeError:
            pass

    def test_model(self, show_progression_bar: bool = True):
        """
        Test model attributes:
        - shape of reward and transition matrices
        - stochasticity of the transition matrix
        """
        # Shape conditions
        assert self.reward_matrix.shape == (
            self.state_dim,
            self.action_dim,
        ), "The shape of the reward matrix is not correct."

        assert (
            len(self.transition_matrix) == self.action_dim
        ), "The transition matrix does not contain action_dim elements."

        for aa in range(self.action_dim):
            assert self.transition_matrix[aa].shape == (
                self.state_dim,
                self.state_dim,
            ), "The shape of the {}-th transition matrix is not correct".format(aa)

        # Stochasticity condition
        _range = trange if show_progression_bar else range

        for ss1 in _range(self.state_dim):
            for aa in range(self.action_dim):
                assert (
                    abs(self.transition_matrix[aa][[ss1], :].sum() - 1) < 1e-6
                ), f"The transition matrix is not stochastic. Problem with state {ss1}, action {aa}, transition {self.transition_matrix[aa][[ss1], :].sum()}."
                for ss2 in range(self.state_dim):
                    assert (
                        self.transition_matrix[aa][ss1, ss2] >= 0.0
                    ), "Negative transition."

    def _normalize_reward_matrix(self):
        """
        Transform the reward matrix of the model :
        0 <= self.reward_matrix <= 1 after application.
        #"""
        if abs(self.reward_matrix.max() - self.reward_matrix.min() - 1) < 1e-4:
            return
        else:
            self.reward_matrix -= self.reward_matrix.min()
            self.reward_matrix /= self.reward_matrix.max()

    def _normalize_transition_sparse(self):
        for aa in range(self.action_dim):
            for ss1 in range(self.state_dim):
                transition_probability = self.transition_matrix[aa].getrow(ss1).sum()
                if transition_probability <= 0:
                    self.transition_matrix[aa][ss1, ss1] = 1.0
                else:
                    inv = 1 / transition_probability
                    for ss2 in range(self.state_dim):
                        self.transition_matrix[aa][ss1, ss2] = (
                            inv * self.transition_matrix[aa][ss1, ss2]
                        )

    def _normalize_transition_numpy(self):
        for aa in range(self.action_dim):
            for ss1 in range(self.state_dim):
                if self.transition_matrix[aa][ss1, :].sum() == 0:
                    self.transition_matrix[aa][ss1, ss1] = 1.0
                else:
                    self.transition_matrix[aa][ss1, :] /= self.transition_matrix[aa][
                        ss1, :
                    ].sum()

    def _normalize_transition_matrix(self):
        if isinstance(self.transition_matrix, list):
            self._normalize_transition_sparse()
        else:
            self._normalize_transition_numpy()

    def get_transition_density(self) -> float:
        return (
            sum(len(matrix.data) for matrix in self.transition_matrix)
            / self.state_dim**2
            / self.action_dim
        )

    def get_reward_density(self) -> float:
        return np.count_nonzero(self.reward_matrix) / self.state_dim / self.action_dim

    def get_model_type(self) -> str:
        if isinstance(self.transition_matrix, list):
            return SPARSE
        else:
            return NUMPY

    def _model_to_numpy(self):
        """Convert the transition and reward matrices to numpy arrays."""
        try:
            self.transition_matrix = np.array(
                [matrix.toarray() for matrix in self.transition_matrix]
            )
        except AttributeError:
            pass
        try:
            self.reward_matrix = self.reward_matrix.toarray()
        except AttributeError:
            pass

    def _model_to_sparse(self):
        self.transition_matrix = [
            csr_array(matrix) for matrix in self.transition_matrix
        ]

    def _model_to_marmote(self):
        """Convert the transition and reward matrices to marmote format."""
        assert self._is_model_built(), "Model is not built yet."

        from utils.numpy_to_marmote import (
            build_marmote_transition_list,
            build_marmote_reward_matrix,
        )

        self.transition_matrix = build_marmote_transition_list(
            self.state_dim,
            self.action_dim,
            self.transition_matrix,
        )
        self.reward_matrix = build_marmote_reward_matrix(
            self.state_dim,
            self.action_dim,
            self.reward_matrix,
        )

    def _is_model_built(self) -> bool:
        return hasattr(self, "transition_matrix") and hasattr(self, "reward_matrix")

    def _convert_model(self, mode: str):
        if mode == NUMPY:
            self._model_to_numpy()
        elif mode == SPARSE:
            self._model_to_sparse()
        else:
            assert False, "Unknown mode {}".format(mode)

    def get_model_main_parameters(self) -> list:
        # State dim, action dim, transition density, transition average, transition std, reward density, reward average, reward std
        result = [self.state_dim, self.action_dim, self.get_transition_density()]
        trans_avg = np.mean([matrix.mean() for matrix in self.transition_matrix])
        trans_std = np.std(
            np.concatenate([matrix.data for matrix in self.transition_matrix])
        )
        reward_density = self.get_reward_density()
        reward_avg = np.mean(self.reward_matrix)
        reward_std = np.std(self.reward_matrix)
        result += [trans_avg, trans_std, reward_density, reward_avg, reward_std]
        return result

    ###### DYNAMIC PROGRAMMING METHODS ######

    def bellman(self, value: np.ndarray, discount: float) -> np.ndarray:
        """
        Returns R + discount * T @ V
        """
        q_value = np.empty((self.action_dim, self.state_dim))
        for aa in range(self.action_dim):
            q_value[aa] = self.reward_matrix[:, aa] + discount * self.transition_matrix[
                aa
            ].dot(value)
        return q_value

    def optimal_bellman(self, value: np.ndarray, discount: float) -> np.ndarray:
        """Compute the optimal Bellman operator for the model."""
        return np.max(self.bellman(value, discount), axis=0)

    def optimal_bellman_q(self, q_value: np.ndarray, discount: float) -> np.ndarray:
        value = q_value.max(axis=0)
        return self.bellman(value, discount)

    def bellman_policy(
        self,
        value: np.ndarray,
        discount: float,
        transition_policy: csr_array,
        reward_policy: np.ndarray,
    ) -> np.ndarray:
        """
        Returns the Bellman operator applied to a value function with a given policy.
        """
        return reward_policy + discount * transition_policy.dot(value)

    ###### UTILITY METHODS ######

    def transition_reward_policy(self, policy: np.ndarray) -> tuple:
        """Compute the transition and reward policy for a given policy."""
        transition_policy = lil_matrix((self.state_dim, self.state_dim))
        reward_policy = np.zeros(self.state_dim)
        for aa in range(self.action_dim):
            ind = (policy == aa).nonzero()[0]
            if ind.size > 0:
                for ss in ind:
                    transition_policy[[ss], :] = self.transition_matrix[aa][
                        [ss], :
                    ].toarray()
                reward_policy[ind] = self.reward_matrix[ind, aa]
        transition_policy = transition_policy.tocsr()
        # model._model_to_numpy()
        return transition_policy, reward_policy

    ###### ALGORITHMS ######

    def iterative_policy_evaluation(
        self,
        policy: np.ndarray,
        discount: float,
        variation_tol: float,
        initial_value: np.ndarray = None,
    ) -> np.ndarray:
        """
        Iterative policy evaluation algorithm.
        """
        if initial_value is None:
            value = np.zeros((self.state_dim))
        else:
            value = initial_value

        transition_policy, reward_policy = self.transition_reward_policy(policy)

        while True:
            next_value = self.bellman_policy(
                value, discount, transition_policy, reward_policy
            )
            distance = norminf(next_value - value)
            value = next_value

            if distance < variation_tol:
                break

        return value

    def value_iteration(self, discount: float, variation_tol: float) -> np.ndarray:
        while True:
            new_value = self.bellman(value, discount)
            variation = norminf(new_value - value)
            value = new_value
            if variation < variation_tol:
                break

        return value, variation
