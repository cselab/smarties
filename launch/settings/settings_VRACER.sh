SETTINGS=
#discount factor in RL
#the closer to 1 it is, the harder it is to learn
#but, the agent might find better long-term strategies
SETTINGS+=" --gamma 0.995"

#size of network layers
SETTINGS+=" --nnl1 128"
SETTINGS+=" --nnl2 128"
#SETTINGS+=" --nnl3 128"

#SETTINGS+=" --nnType RNN"
#SETTINGS+=" --nnFunc PRelu"
SETTINGS+=" --nnFunc SoftSign"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"
#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learner VRACER"
#chance of taking random actions
SETTINGS+=" --explNoise 0.5"

SETTINGS+=" --obsPerStep 1"
SETTINGS+=" --minTotObsNum 131072"
SETTINGS+=" --maxTotObsNum 524288"

SETTINGS+=" --totNumSteps 5000000"
SETTINGS+=" --clipImpWeight 4"

SETTINGS+=" --penalTol 0.1"
SETTINGS+=" --targetDelay 0"
#batch size for network gradients compute
SETTINGS+=" --batchSize 256"
SETTINGS+=" --bSampleSequences 0"
#network update learning rate
SETTINGS+=" --learnrate 0.0001"
