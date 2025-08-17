import marmote.core as mc
import marmote.mdp as mmdp

actionSpace = mc.MarmoteInterval(0, 1)
stateSpace = mc.MarmoteInterval(0, 1)

trans = list()
P0 = mc.SparseMatrix(2)

P0.setEntry(0, 0, 0.6)
P0.setEntry(0, 1, 0.4)
P0.setEntry(1, 0, 0.5)
P0.setEntry(1, 1, 0.5)

print("Matrix", P0)

trans.append(P0)

P1 = mc.SparseMatrix(2)
P1.setEntry(0, 0, 0.2)
P1.setEntry(0, 1, 0.8)
P1.setEntry(1, 0, 0.7)
P1.setEntry(1, 1, 0.3)
trans.append(P1)

Reward = mc.FullMatrix(2, 2)

Reward.setEntry(0, 0, 4.5)
Reward.setEntry(0, 1, 2)
Reward.setEntry(1, 0, -1.5)
Reward.setEntry(1, 1, 3)

beta = 0.95
criterion = "max"
mdp = mmdp.DiscountedMDP(criterion, stateSpace, actionSpace, trans, Reward, beta)
print(mdp)

epsilon = 0.00001
maxIter = 700

optimum = mdp.ValueIteration(epsilon, maxIter)

print(optimum)

optimum2 = mdp.ValueIterationGS(epsilon, 10)
print(optimum2)

optimum3 = mdp.ValueIterationInit(epsilon, 200, optimum2)
print("Optimum 3", optimum3)

optimum4 = mdp.PolicyIterationModified(epsilon, maxIter, 0.001, 100)
print("optimum4", optimum4)

optimum5 = mdp.PolicyIterationModifiedGS(epsilon, maxIter, 0.001, 100)
print("last test", optimum5)
