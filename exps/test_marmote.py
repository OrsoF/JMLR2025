from models.cliff import Model

model = Model(400, 4)
model.create_model()
model._model_to_marmote()

trans = model.transition_matrix
rew = model.reward_matrix

import marmote.core as mc
import marmote.mdp as mmdp

actionSpace = mc.MarmoteInterval(0, model.action_dim - 1)
stateSpace = mc.MarmoteInterval(0, model.state_dim - 1)


beta = 0.95
criterion = "max"

epsilon = 0.00001
maxIter = 700

mdp = mmdp.DiscountedMDP(criterion, stateSpace, actionSpace, trans, Reward, beta)
optimum = mdp.ValueIteration(epsilon, maxIter)
print(optimum.getValue())
