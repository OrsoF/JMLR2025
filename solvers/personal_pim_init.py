import numpy as np
from time import time
from utils.generic_model import GenericModel, NUMPY, SPARSE
from scipy.sparse import lil_matrix


class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
        mode: str = SPARSE,
    ):
        # Class arguments
        self.model = model
        self.discount = discount
        self.epsilon_policy_evaluation = final_precision
        self.mode = mode

        self.name = "PIM"

        self.model._convert_model(self.mode)
        self.max_iter_evaluation = int(1e8)
        self.max_iter_policy_update = int(1e8)

        # print("Retirer le tocsr dans Policy Iteration modified.")

    def run(self):
        start_time = time()
        self.policy = np.zeros((self.model.state_dim))
        self.policy = np.argmax(self.model.reward_matrix, axis=1)
        # self.value = np.zeros((self.model.state_dim))
        self.value = self.model.reward_matrix.max(axis=1)

        policy_update_iter = 0
        while True:
            self.value = self._policy_evaluation(
                self.policy,
                self.epsilon_policy_evaluation,
                self.max_iter_evaluation,
                self.value,
            )
            new_policy = self.bellman_no_max(self.value).argmax(axis=1)

            policy_update_condition = np.all(new_policy == self.policy)
            max_iter_condition = policy_update_iter == self.max_iter_policy_update

            if policy_update_condition or max_iter_condition:
                self.runtime = time() - start_time
                break
            else:
                self.policy = new_policy

    def _policy_evaluation(
        self,
        policy: np.ndarray,
        epsilon_policy_evaluation: float,
        max_iteration_evaluation: int,
        value: np.ndarray,
    ) -> np.ndarray:
        eval_iter = 0
        transition_policy, reward_policy = self._compute_transition_reward_pi(policy)

        while True:
            eval_iter += 1
            new_value = reward_policy + self.discount * transition_policy.dot(value)
            variation = np.absolute(new_value - value).max()
            if (
                variation
                < ((1 - self.discount) / self.discount) * epsilon_policy_evaluation
            ) or eval_iter == max_iteration_evaluation:
                return new_value
            else:
                value = new_value

    def _compute_transition_reward_pi(self, policy):
        if self.mode == NUMPY:
            transition_policy = np.empty((self.model.state_dim, self.model.state_dim))
            reward_policy = np.zeros(self.model.state_dim)
            for aa in range(self.model.action_dim):
                ind = (policy == aa).nonzero()[0]
                # if no rows use action a, then no need to assign this
                if ind.size > 0:
                    transition_policy[ind, :] = self.model.transition_matrix[aa][ind, :]
                    reward_policy[ind] = self.model.reward_matrix[ind, aa]
            return transition_policy, reward_policy
        elif self.mode == SPARSE:

            # Methode 1
            # transition_policy = lil_matrix((self.model.state_dim, self.model.state_dim))
            # reward_policy = np.zeros(self.model.state_dim)
            # for aa in range(self.model.action_dim):
            #     ind: np.ndarray = (policy == aa).nonzero()[0]
            #     if ind.size > 0:
            #         # transition_policy[ind, :] = model.transition_matrix[aa][
            #         #     ind, :
            #         # ].toarray()
            #         for i, elem in enumerate(self.model.transition_matrix[aa][ind, :]):
            #             if elem.toarray()[0, 0] > 0:
            #                 transition_policy[ind, i] = elem
            # # transition_policy = transition_policy.tocsr()
            # return transition_policy, reward_policy

            # Methode 2
            transition_policy = lil_matrix((self.model.state_dim, self.model.state_dim))
            reward_policy = np.zeros(self.model.state_dim)
            for aa in range(self.model.action_dim):
                ind = (policy == aa).nonzero()[0]
                if ind.size > 0:
                    for ss in ind:
                        transition_policy[[ss], :] = self.model.transition_matrix[aa][
                            [ss], :
                        ].toarray()
                    reward_policy[ind] = self.model.reward_matrix[ind, aa]
            transition_policy = transition_policy.tocsr()
            return transition_policy, reward_policy
        else:
            assert False, "Should not be seen."

    def bellman_no_max(self, value):
        q_value = np.zeros((self.model.state_dim, self.model.action_dim))

        for aa in range(self.model.action_dim):
            q_value[:, aa] = self.model.reward_matrix[
                :, aa
            ] + self.discount * self.model.transition_matrix[aa].dot(value)

        return q_value
