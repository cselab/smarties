SETTINGS=
#discount factor in RL
#the closer to 1 it is, the harder it is to learn
#but, the agent might find better long-term strategies
SETTINGS+=" --gamma 0.995"

#size of network layers
#SETTINGS+=" --nnl1 512"
#SETTINGS+=" --nnl2 512"
SETTINGS+=" --nnl1 128"
SETTINGS+=" --nnl2 128"
#SETTINGS+=" --nnl3 128"

#subject to changes
#SETTINGS+=" --nnFunc PRelu"
#SETTINGS+=" --nnFunc Tanh"
SETTINGS+=" --nnFunc SoftSign"
#SETTINGS+=" --nnFunc HardSign"
SETTINGS+=" --outWeightsPrefac 0.1"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"

#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learner RACER"

SETTINGS+=" --maxTotObsNum ${BUFFSIZE}"
SETTINGS+=" --clipImpWeight ${IMPSAMPR}"

#chance of taking random actions
SETTINGS+=" --explNoise 1"
SETTINGS+=" --totNumSteps 10000000"
SETTINGS+=" --obsPerStep ${EPERSTEP}"
SETTINGS+=" --bSampleSequences 0"

#lag of target network.
#- if >1 (ie 1000) then weights are copied every dqnT grad descent steps
#- if <1 (ie .001) then every step the target weights are updated as dqnT * w_Target + (1-dqnT)*w
SETTINGS+=" --targetDelay 0"
SETTINGS+=" --penalTol ${TOLPARAM}"
#batch size for network gradients compute
SETTINGS+=" --batchSize ${BATCHNUM}"
#network update learning rate
SETTINGS+=" --learnrate 0.0001"
