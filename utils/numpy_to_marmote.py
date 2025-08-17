"""
Function that build a marmote MDP model from python matrices.
"""

from marmote.core import MarmoteInterval, SparseMatrix, FullMatrix
from marmote.mdp import (
    GenericMDP,
    DiscountedMDP,
    FeedbackSolutionMDP,
    TotalRewardMDP,
    AverageMDP,
)
import numpy as np
from typing import Optional


AVERAGE = "average"
DISCOUNTED = "discounted"
TOTAL = "total"


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


def _build_transition_matrix(self):
    self.transitions_list = list()
    for aa in range(self.env.action_dim):
        P = SparseMatrix(self.env.state_dim)

        row_indices, col_indices = self.env.transition_matrix[aa].nonzero()
        nonzero_values = self.env.transition_matrix[aa][row_indices, col_indices]

        for ss1, ss2, value in zip(row_indices, col_indices, nonzero_values):
            P.addToEntry(int(ss1), int(ss2), value)

        # self.mdp.addMatrix(a, P)
        self.transitions_list.append(P)
        P = None


def build_marmote_model(
    transition_matrix: list,
    reward_matrix: np.ndarray,
    criterion: str,
    discount: Optional[float] = None,
) -> GenericMDP:
    """
    Input :
    transition_matrix : liste de taille A de matrices de shape SxS
    reward_matrix : matrice de taille SxA
    discount :
    """
    state_dim, action_dim = reward_matrix.shape
    state_space = MarmoteInterval(0, int(state_dim - 1))
    action_space = MarmoteInterval(0, int(action_dim - 1))

    marmote_reward_matrix = build_marmote_reward_matrix(
        state_dim, action_dim, reward_matrix
    )

    marmote_transition_list = build_marmote_transition_list(
        state_dim, action_dim, transition_matrix
    )

    if criterion == AVERAGE:
        mdp = AverageMDP(
            "max",
            state_space,
            action_space,
            marmote_transition_list,
            marmote_reward_matrix,
        )
    elif criterion == DISCOUNTED:
        mdp = DiscountedMDP(
            "max",
            state_space,
            action_space,
            marmote_transition_list,
            marmote_reward_matrix,
            discount,
        )
    elif criterion == TOTAL:
        mdp = TotalRewardMDP(
            "max",
            state_space,
            action_space,
            marmote_transition_list,
            marmote_reward_matrix,
        )
    else:
        assert False, "Criterion {} not recognized. Choose {} or {} or {}".format(
            criterion, DISCOUNTED, TOTAL, AVERAGE
        )

    return mdp
