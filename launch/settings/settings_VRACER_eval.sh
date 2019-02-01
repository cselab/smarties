SETTINGS=
#discount factor in RL
#the closer to 1 it is, the harder it is to learn
#but, the agent might find better long-term strategies
SETTINGS+=" --gamma 0.995  --samplesFile 1"

#size of network layers
SETTINGS+=" --nnl1 128"
SETTINGS+=" --nnl2 128"
#SETTINGS+=" --nnl3 128"

#SETTINGS+=" --nnType RNN"
#SETTINGS+=" --nnFunc LRelu"
SETTINGS+=" --nnFunc SoftSign"
SETTINGS+=" --outWeightsPrefac 0.1"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 0"
#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learner VRACER"
#chance of taking random actions
SETTINGS+=" --explNoise 0.0"

SETTINGS+=" --obsPerStep 1"
SETTINGS+=" --minTotObsNum 262144"
SETTINGS+=" --maxTotObsNum 262144"

SETTINGS+=" --totNumSteps 1"
SETTINGS+=" --clipImpWeight 4"

SETTINGS+=" --penalTol 0.1"
SETTINGS+=" --targetDelay 0"
#batch size for network gradients compute
SETTINGS+=" --batchSize 256"
SETTINGS+=" --bSampleSequences 0"
SETTINGS+=" --ERoldSeqFilter oldest"
#SETTINGS+=" --dataSamplingAlgo PERseq"
#network update learning rate
SETTINGS+=" --learnrate 0.000"
