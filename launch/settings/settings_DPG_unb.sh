SETTINGS=
#discount factor in RL
#the closer to 1 it is, the harder it is to learn
#but, the agent might find better long-term strategies
SETTINGS+=" --gamma 0.995"

#size of network layers
SETTINGS+=" --nnl1 128"
SETTINGS+=" --nnl2 128"

#subject to changes
#SETTINGS+=" --nnType RNN"
#SETTINGS+=" --nnType LSTM"
SETTINGS+=" --nnFunc SoftSign"
# Multiplies initial weights of output layer. Ie U[-.1/sqrt(f), .1/sqrt(f)]
SETTINGS+=" --outWeightsPrefac 0.1"

#whether you are training a policy or testing an already trained network
SETTINGS+=" --bTrain 1"
SETTINGS+=" --epsAnneal 0"

#RL algorithm: NAF, DPG are continuous actions, NFQ (also accepted DQN) is for discrete actions
SETTINGS+=" --learner DPG"

SETTINGS+=" --maxTotObsNum 524288"
SETTINGS+=" --minTotObsNum 131072"
#SETTINGS+=" --minTotObsNum 35536"

#chance of taking random actions
SETTINGS+=" --explNoise 0.2"
SETTINGS+=" --totNumSteps 10000000"
SETTINGS+=" --obsPerStep 1"
SETTINGS+=" --bSampleSequences 0"
#SETTINGS+=" --impWeight 4"
SETTINGS+=" --clipImpWeight 0"
SETTINGS+=" --penalTol 0.1"

#lag of target network.
#- if >1 (ie 1000) then weights are copied every dqnT grad descent steps
#- if <1 (ie .001) then every step the target weights are updated as dqnT * w_Target + (1-dqnT)*w
#the first option is markedly safer
SETTINGS+=" --targetDelay 0.01"
#batch size for network gradients compute
SETTINGS+=" --batchSize 128"
#network update learning rate
#SETTINGS+=" --learnrate 0.00001"
SETTINGS+=" --learnrate 0.00001"
