SETTINGS=
#discount factor in RL
#the closer to 1 it is, the harder it is to learn
#but, the agent might find better long-term strategies
SETTINGS+=" --gamma 0.995"

#size of network layers
SETTINGS+=" --nnl1 ${LAYRSIZE}"
SETTINGS+=" --nnl2 ${LAYRSIZE}"
#SETTINGS+=" --nnl3 128"

#subject to changes
#SETTINGS+=" --nnFunc LRelu"
#SETTINGS+=" --nnFunc Tanh"
SETTINGS+=" --nnFunc SoftSign"
#SETTINGS+=" --nnFunc HardSign"
SETTINGS+=" --outWeightsPrefac 0.1"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"

#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learner RACER"
#SETTINGS+=" --learner DACER"

#SETTINGS+=" --minTotObsNum  32768"
SETTINGS+=" --minTotObsNum 131072"
SETTINGS+=" --maxTotObsNum ${BUFFSIZE}"
SETTINGS+=" --clipImpWeight ${IMPSAMPR}"

#chance of taking random actions
SETTINGS+=" --explNoise 0.5"
SETTINGS+=" --epsAnneal 0"
SETTINGS+=" --totNumSteps 10000000"
#SETTINGS+=" --totNumSteps 5000000"
SETTINGS+=" --obsPerStep ${EPERSTEP}"

#lag of target network.
#- if >1 (ie 1000) then weights are copied every dqnT grad descent steps
#- if <1 (ie .001) then every step the target weights are updated as dqnT * w_Target + (1-dqnT)*w
SETTINGS+=" --targetDelay 0"
SETTINGS+=" --penalTol ${TOLPARAM}"
#batch size for network gradients compute
SETTINGS+=" --batchSize ${BATCHNUM}"
#network update learning rate
SETTINGS+=" --learnrate ${NETLRATE}"
