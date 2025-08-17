import sys
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(color_scheme="Linux", call_pdb=False)


import marmote.core as mco
import marmote.markovchain as mch
import marmote.mdp as md
import numpy as np

model = dict()
model["B1"] = 3  # 10
model["B2"] = 3  # 10
model["K1"] = 2  # 5
model["K2"] = 2  # 5
model["lam"] = 5.0
model["mu1"] = 1.0
model["mu2"] = 1.0
model["Ca"] = 1
model["Cd"] = 1
model["Cr"] = 10
model["Cs"] = 2
model["Ch"] = 2
model["beta"] = 1

print(model)

dims = np.array([model["B1"], model["K1"], model["B2"], model["K2"]])
print(dims)
states = mco.MarmoteBox(dims)
#
actions = mco.MarmoteBox([model["K1"], model["K2"]])

print("Number of states", states.Cardinal())
print(states)
print("Number of actions", actions.Cardinal())
print("actions", actions)


def fill_in_matrix(
    index_action: int, modele: dict, ssp: mco.MarmoteBox, asp: mco.MarmoteBox
):
    # retrieve the action asscoiated with index
    action_buf = asp.DecodeState(index_action)
    # *#print("index action",index_action,"action",action_buf)
    # define the states
    etat = np.array([0, 0, 0, 0])
    afteraction = np.array([0, 0, 0, 0])
    jump = np.array([0, 0, 0, 0])
    # define transition matrix
    P = mco.SparseMatrix(ssp.Cardinal())
    # browsing state space
    ssp.FirstState(etat)
    for k in range(ssp.Cardinal()):
        # compute the index of the state
        indexL = ssp.Index(etat)
        # compute the state after the action
        afteraction[0] = etat[0]
        afteraction[1] = action_buf[0]
        afteraction[2] = etat[2]
        afteraction[3] = action_buf[1]
        # *# print("####index State=",k,"State",etat,"State after action",afteraction)
        # then detail all the possible transitions
        ## Arrival (increases the number of customer in first coordinate with rate lambda)
        if afteraction[0] < modele["B1"] - 1:
            jump[0] = afteraction[0] + 1
            jump[1] = afteraction[1]
            jump[2] = afteraction[2]
            jump[3] = afteraction[3]
        else:
            jump[0] = afteraction[0]
            jump[1] = afteraction[1]
            jump[2] = afteraction[2]
            jump[3] = afteraction[3]
        # compute the index of the jump
        indexC = ssp.Index(jump)
        # fill in the entry
        # *# print("*Event: Arrival. Index=",indexC,"Jump State=",jump,"rate=",modele['lam'])
        P.setEntry(indexL, indexC, modele["lam"])
        #
        ## departure of the first system entry in the second one
        if afteraction[2] < modele["B2"] - 1:
            jump[0] = max(0, afteraction[0] - 1)
            jump[1] = afteraction[1]
            jump[2] = afteraction[2] + 1
            jump[3] = afteraction[3]
        else:
            jump[0] = max(0, afteraction[0] - 1)
            jump[1] = afteraction[1]
            jump[2] = afteraction[2]
            jump[3] = afteraction[3]
        # index of the jump
        indexC = ssp.Index(jump)
        # rate of the transition
        rate = min(afteraction[1], afteraction[0]) * modele["mu1"]
        # fill in the entry
        # *# print("*Event: Departure s1 entry s2. Index=",indexC,"Jump State=",jump,"rate=",rate)
        P.setEntry(indexL, indexC, rate)
        #
        ##departure of the second  system
        jump[0] = afteraction[0]
        jump[1] = afteraction[1]
        jump[2] = max(0, afteraction[2] - 1)
        jump[3] = afteraction[3]
        # compute the index of the jump
        indexC = ssp.Index(jump)
        # compute the rate
        rate = min(afteraction[2], afteraction[3]) * modele["mu2"]
        # fill in the entry
        # *# print("*Event: Departure s2. Index=",indexC,"Jump State=",jump,"rate=",rate)
        P.setEntry(indexL, indexC, rate)
        # change state
        ssp.NextState(etat)
    return P


def fill_in_cost(modele, ssp, asp):
    R = mco.FullMatrix(ssp.Cardinal(), asp.Cardinal())
    # define the states
    etat = np.array([0, 0, 0, 0])
    # define the actions
    acb = asp.StateBuffer()
    ssp.FirstState(etat)
    for k in range(ssp.Cardinal()):
        # compute the index of the state
        indexL = ssp.Index(etat)
        # *#print("##State",etat)
        asp.FirstState(acb)
        for j in range(asp.Cardinal()):
            # *#print("---Action",acb,end='  ')
            action1 = acb[0]
            action2 = acb[1]
            totalrate = (
                modele["lam"] + action1 * modele["mu1"] + action2 * modele["mu2"]
            )
            activationcosts = modele["Ca"] * (
                max(0, action1 - etat[1]) + max(0, action2 - etat[3])
            )
            deactivationcosts = modele["Cd"] * (
                max(0, etat[1] - action1) + max(0, action2 - etat[3])
            )
            rejectioncosts = 0.0
            if (modele["B1"] - 1) == etat[0]:
                rejectioncosts += (modele["lam"] * modele["Cr"]) / totalrate
            if (modele["B2"] - 1) == etat[2]:
                rejectioncosts += (
                    min(etat[0], action1) * modele["mu1"] * modele["Cr"]
                ) / totalrate
            instantaneouscosts = activationcosts + deactivationcosts + rejectioncosts
            accumulatedcosts = (etat[0] + etat[2]) * modele["Ch"] + (
                action1 + action2
            ) * modele["Cs"]
            accumulatedcosts /= totalrate + model["beta"]
            # *#print("Instantaneous=",instantaneouscosts," Rejection=",rejectioncosts,end= ' ')
            # *#print("Accumulatedcosts=",accumulatedcosts)
            R.setEntry(indexL, j, accumulatedcosts + instantaneouscosts)
            asp.NextState(acb)
        ssp.NextState(etat)
    return R


trans = list()

action_buf = actions.StateBuffer()
actions.FirstState(action_buf)
for k in range(actions.Cardinal()):
    trans.append(fill_in_matrix(k, model, states, actions))
    print("---Matrix kth=", k, "filled in")

print("Matrice of Costs")
Costs = fill_in_cost(model, states, actions)

print("Begining of Building MDP")
ctmdp = md.ContinuousTimeDiscountedMDP(
    "min", states, actions, trans, Costs, model["beta"]
)
print(ctmdp)

ctmdp.UniformizeMDP()
print("Rate of Uniformization", ctmdp.getMaximumRate())
# *# print(ctmdp)
policy = md.FeedbackSolutionMDP(states.Cardinal())

etat = states.StateBuffer()
states.FirstState(etat)
for k in range(states.Cardinal()):
    if etat[0] == (model["B1"] - 1) or etat[2] == (model["B2"] - 1):
        policy.setActionIndex(k, 0)
    else:
        policy.setActionIndex(k, 1)
    states.NextState(etat)
print(policy)

Mat = ctmdp.GetChain(optimum)
Mat.set_type(mco.DISCRETE)
# *# print(Mat)

initial = mco.UniformDiscreteDistribution(0, states.Cardinal() - 1)
