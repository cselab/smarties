//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include <getopt.h>
#include "Utils/ArgParser.h"
#include "Utils/Warnings.h"

struct Settings
{
  Settings() {}
  ~Settings() {}

//To modify from default value any of these settings, run executable with either
//- ascii symbol of the setting (CHARARG) followed by the value (ie. -# $value)
//- the name of the setting variable followed by the value (ie. -setting $value)

///////////////////////////////////////////////////////////////////////////////
//SETTINGS PERTAINING TO LEARNING ALGORITHM: lowercase LETTER
///////////////////////////////////////////////////////////////////////////////
#define CHARARG_learner 'a'
#define COMMENT_learner "Algorithm."
#define TYPEVAL_learner string
#define TYPENUM_learner STRING
#define DEFAULT_learner "RACER"
  string learner = DEFAULT_learner;

#define CHARARG_bTrain 'b'
#define COMMENT_bTrain "Whether training a policy (=1) or evaluating (=0)."
#define TYPEVAL_bTrain int
#define TYPENUM_bTrain INT
#define DEFAULT_bTrain 1
  int bTrain = DEFAULT_bTrain;

#define CHARARG_clipImpWeight 'c'
#define COMMENT_clipImpWeight "Max importance weight for off-policy Policy \
Gradient. Algo specific."
#define TYPEVAL_clipImpWeight Real
#define TYPENUM_clipImpWeight REAL
#define DEFAULT_clipImpWeight 4
Real clipImpWeight = DEFAULT_clipImpWeight;

#define CHARARG_targetDelay 'd'
#define COMMENT_targetDelay "Copy delay for target network. If >1: every \
$targetDelay grad desc steps tgt-net copies curr weigths. If <1: every \
grad desc step tgt-net does exp averaging."
#define TYPEVAL_targetDelay Real
#define TYPENUM_targetDelay REAL
#define DEFAULT_targetDelay 0
  Real targetDelay = DEFAULT_targetDelay;

#define CHARARG_explNoise 'e'
#define COMMENT_explNoise "Noise added to policy. For discrete actions \
it is the probability of picking a random one (detail depend on chosen \
learning algorithm), for continuous actions it is the (initial) stdev."
#define TYPEVAL_explNoise Real
#define TYPENUM_explNoise REAL
#define DEFAULT_explNoise 0.5
  Real explNoise = DEFAULT_explNoise;

#define CHARARG_ERoldSeqFilter 'f'
#define COMMENT_ERoldSeqFilter "Filter algorithm to remove old episodes from \
memory buffer. Accepts: oldest, farpolfrac, maxkldiv, minerror, or default. \
Default means oldest for ER and farpolfrac for ReFER"
#define TYPEVAL_ERoldSeqFilter string
#define TYPENUM_ERoldSeqFilter STRING
#define DEFAULT_ERoldSeqFilter "default"
  string ERoldSeqFilter = DEFAULT_ERoldSeqFilter;

#define CHARARG_gamma 'g'
#define COMMENT_gamma "Discount factor."
#define TYPEVAL_gamma Real
#define TYPENUM_gamma REAL
#define DEFAULT_gamma 0.995
  Real gamma = DEFAULT_gamma;

#define CHARARG_klDivConstraint 'k'
#define COMMENT_klDivConstraint "Constraint on max KL div, algo specific."
#define TYPEVAL_klDivConstraint Real
#define TYPENUM_klDivConstraint REAL
#define DEFAULT_klDivConstraint 0.01
  Real klDivConstraint = DEFAULT_klDivConstraint;

#define CHARARG_lambda 'l'
#define COMMENT_lambda "Lambda for off policy corrections."
#define TYPEVAL_lambda Real
#define TYPENUM_lambda REAL
#define DEFAULT_lambda 0.95
  Real lambda = DEFAULT_lambda;

#define CHARARG_minTotObsNum 'm'
#define COMMENT_minTotObsNum "Min number of transitions in training buffer \
before training starts. If unset we use maxTotObsNum."
#define TYPEVAL_minTotObsNum int
#define TYPENUM_minTotObsNum INT
#define DEFAULT_minTotObsNum -1
  int minTotObsNum = DEFAULT_minTotObsNum;

#define CHARARG_maxTotObsNum 'n'
#define COMMENT_maxTotObsNum "Max number of transitions in training buffer."
#define TYPEVAL_maxTotObsNum int
#define TYPENUM_maxTotObsNum INT
#define DEFAULT_maxTotObsNum 1000000
  int maxTotObsNum = DEFAULT_maxTotObsNum;

#define CHARARG_obsPerStep 'o'
#define COMMENT_obsPerStep "Ratio of observed *transitions* to gradient \
steps. 0.1 means that for every observation, learner does 10 gradient steps."
#define TYPEVAL_obsPerStep  Real
#define TYPENUM_obsPerStep  REAL
#define DEFAULT_obsPerStep  1
  Real obsPerStep = DEFAULT_obsPerStep;

#define CHARARG_penalTol 't'
#define COMMENT_penalTol "Tolerance used for adaptive penalization methods. \
Algo specific."
#define TYPEVAL_penalTol  Real
#define TYPENUM_penalTol  REAL
#define DEFAULT_penalTol  0.1
  Real penalTol = DEFAULT_penalTol;

#define CHARARG_epsAnneal 'r'
#define COMMENT_epsAnneal "Annealing rate in grad steps of various \
learning-algorithm-dependent behaviors."
#define TYPEVAL_epsAnneal Real
#define TYPENUM_epsAnneal REAL
#define DEFAULT_epsAnneal 5e-7
  Real epsAnneal = DEFAULT_epsAnneal;

#define CHARARG_bSampleSequences 's'
#define COMMENT_bSampleSequences "Whether to sample sequences (1) \
or observations (0) from the Replay Memory."
#define TYPEVAL_bSampleSequences  int
#define TYPENUM_bSampleSequences  INT
#define DEFAULT_bSampleSequences  0
  int bSampleSequences = DEFAULT_bSampleSequences;

#define CHARARG_bSharedPol 'y'
#define COMMENT_bSharedPol "Have a separate policy per each agent on a sim"
#define TYPEVAL_bSharedPol int
#define TYPENUM_bSharedPol INT
#define DEFAULT_bSharedPol 1
  int bSharedPol = DEFAULT_bSharedPol;

#define CHARARG_totNumSteps 'z'
#define COMMENT_totNumSteps "Number of gradient steps before end of learning"
#define TYPEVAL_totNumSteps int
#define TYPENUM_totNumSteps INT
#define DEFAULT_totNumSteps 10000000
  int totNumSteps = DEFAULT_totNumSteps;

///////////////////////////////////////////////////////////////////////////////
//SETTINGS PERTAINING TO NETWORK: CAPITAL LETTER
///////////////////////////////////////////////////////////////////////////////
#define CHARARG_nnl1 'Z'
#define COMMENT_nnl1 "Size of first non-convolutional layer (LSTM/RNN/FFNN)."
#define TYPEVAL_nnl1 int
#define TYPENUM_nnl1 INT
#define DEFAULT_nnl1 0
  int nnl1 = DEFAULT_nnl1;

#define CHARARG_nnl2 'Y'
#define COMMENT_nnl2 "Size of second non-convolutional layer (LSTM/RNN/FFNN)."
#define TYPEVAL_nnl2 int
#define TYPENUM_nnl2 INT
#define DEFAULT_nnl2 0
  int nnl2 = DEFAULT_nnl2;

#define CHARARG_nnl3 'X'
#define COMMENT_nnl3 "Size of third non-convolutional layer (LSTM/RNN/FFNN)."
#define TYPEVAL_nnl3 int
#define TYPENUM_nnl3 INT
#define DEFAULT_nnl3 0
  int nnl3 = DEFAULT_nnl3;

#define CHARARG_nnl4 'W'
#define COMMENT_nnl4 "Size of fourth non-convolutional layer (LSTM/RNN/FFNN)."
#define TYPEVAL_nnl4 int
#define TYPENUM_nnl4 INT
#define DEFAULT_nnl4 0
  int nnl4 = DEFAULT_nnl4;

#define CHARARG_nnl5 'V'
#define COMMENT_nnl5 "Size of fifth non-convolutional layer (LSTM/RNN/FFNN)."
#define TYPEVAL_nnl5 int
#define TYPENUM_nnl5 INT
#define DEFAULT_nnl5 0
  int nnl5 = DEFAULT_nnl5;

#define CHARARG_nnl6 'U'
#define COMMENT_nnl6 "Size of sixth non-convolutional layer (LSTM/RNN/FFNN)."
#define TYPEVAL_nnl6 int
#define TYPENUM_nnl6 INT
#define DEFAULT_nnl6 0
  int nnl6 = DEFAULT_nnl6;

#define CHARARG_batchSize 'B'
#define COMMENT_batchSize "Network training batch size."
#define TYPEVAL_batchSize int
#define TYPENUM_batchSize INT
#define DEFAULT_batchSize 128
  int batchSize = DEFAULT_batchSize;

#define CHARARG_appendedObs 'C'
#define COMMENT_appendedObs "Number of past observations to be chained \
together to form policy input (eg. see frames in DQN paper)."
#define TYPEVAL_appendedObs int
#define TYPENUM_appendedObs INT
#define DEFAULT_appendedObs 0
  int appendedObs = DEFAULT_appendedObs;

#define CHARARG_nnPdrop 'D'
#define COMMENT_nnPdrop "Unused currently (dropout)."
#define TYPEVAL_nnPdrop Real
#define TYPENUM_nnPdrop REAL
#define DEFAULT_nnPdrop 0
  Real nnPdrop = DEFAULT_nnPdrop;

#define CHARARG_nnOutputFunc 'E'
#define COMMENT_nnOutputFunc "Activation function for output layers."
#define TYPEVAL_nnOutputFunc string
#define TYPENUM_nnOutputFunc STRING
#define DEFAULT_nnOutputFunc "Linear"
  string nnOutputFunc = DEFAULT_nnOutputFunc;

#define CHARARG_nnFunc 'F'
#define COMMENT_nnFunc "Activation function for non-output layers (which should\
 always be linear) which are built from settings. (Relu, Tanh, Sigm, PRelu, \
softSign, softPlus, ...)"
#define TYPEVAL_nnFunc string
#define TYPENUM_nnFunc STRING
#define DEFAULT_nnFunc "SoftSign"
  string nnFunc = DEFAULT_nnFunc;

#define CHARARG_learnrate 'L'
#define COMMENT_learnrate "Learning rate."
#define TYPEVAL_learnrate Real
#define TYPENUM_learnrate REAL
#define DEFAULT_learnrate 1e-4
  Real learnrate = DEFAULT_learnrate;

#define CHARARG_nnType 'N'
#define COMMENT_nnType "Type of non-output layers read from settings. (RNN, \
LSTM, everything else maps to FFNN). Conv2D layers need to be built in \
environment directly."
#define TYPEVAL_nnType string
#define TYPENUM_nnType STRING
#define DEFAULT_nnType "FFNN"
  string nnType = DEFAULT_nnType;

#define CHARARG_outWeightsPrefac 'O'
#define COMMENT_outWeightsPrefac "Output weights initialization factor (will \
be multiplied by default fan-in factor). Picking 1 leads to treating \
output layers with normal Xavier initialization."
#define TYPEVAL_outWeightsPrefac Real
#define TYPENUM_outWeightsPrefac REAL
#define DEFAULT_outWeightsPrefac 1
  Real outWeightsPrefac = DEFAULT_outWeightsPrefac;

#define CHARARG_nnLambda 'P'
#define COMMENT_nnLambda "Penalization factor for network weights. It will be \
multiplied by learn rate: w -= eta * nnLambda * w . L1 decay option in Bund.h"
#define TYPEVAL_nnLambda Real
#define TYPENUM_nnLambda REAL
#define DEFAULT_nnLambda numeric_limits<Real>::epsilon()
  Real nnLambda = DEFAULT_nnLambda;

#define CHARARG_nnBPTTseq 'T'
#define COMMENT_nnBPTTseq "Number of previous steps considered by RNN."
#define TYPEVAL_nnBPTTseq int
#define TYPENUM_nnBPTTseq INT
#define DEFAULT_nnBPTTseq 16
  int nnBPTTseq = DEFAULT_nnBPTTseq;

#define CHARARG_splitLayers 'S'
#define COMMENT_splitLayers "Number of split layers, description in Settings.h"
//"For each output required by algorithm (ie. value, policy, std, ...) " \/
//"how many non-conv layers should be devoted only to one o the outputs. " \/
//"For example if there 2 FF layers of size Z and Y and this arg is set to 1,"\/
//" then each of the outputs is connected to a separate layer of size Y. " \/
//"Each of the Y-size layers are then connected to the first layer of size Z."
#define TYPEVAL_splitLayers int
#define TYPENUM_splitLayers INT
#define DEFAULT_splitLayers 0
  int splitLayers = DEFAULT_splitLayers;

///////////////////////////////////////////////////////////////////////////////
//SETTINGS PERTAINING TO PARALLELIZATION/COMMUNICATION: ASCII SYMBOL
///////////////////////////////////////////////////////////////////////////////
#define CHARARG_nThreads '#'
#define COMMENT_nThreads "Number of threads from threaded training on each \
master rank."
#define TYPEVAL_nThreads int
#define TYPENUM_nThreads INT
#define DEFAULT_nThreads 1
  int nThreads = DEFAULT_nThreads;

#define CHARARG_nMasters '$'
#define COMMENT_nMasters "Number of master ranks (policy-updating ranks)."
#define TYPEVAL_nMasters int
#define TYPENUM_nMasters INT
#define DEFAULT_nMasters 1
  int nMasters = DEFAULT_nMasters;

#define CHARARG_isServer '!'
#define COMMENT_isServer "DEPRECATED: Whether smarties launches environment \
app (=1) or is launched by it (=0) (then cannot train)."
#define TYPEVAL_isServer int
#define TYPENUM_isServer INT
#define DEFAULT_isServer 1
  int isServer = DEFAULT_isServer;

#define CHARARG_ppn '>'
#define COMMENT_ppn "Number of processes per node."
#define TYPEVAL_ppn int
#define TYPENUM_ppn INT
#define DEFAULT_ppn 1
  int ppn = DEFAULT_ppn;

#define CHARARG_sockPrefix '@'
#define COMMENT_sockPrefix "Prefix for communication file over sockets."
#define TYPEVAL_sockPrefix int
#define TYPENUM_sockPrefix INT
#define DEFAULT_sockPrefix 0
  int sockPrefix = DEFAULT_sockPrefix;

#define CHARARG_samplesFile '('
#define COMMENT_samplesFile "Whether to write files recording all transitions."
#define TYPEVAL_samplesFile int
#define TYPENUM_samplesFile INT
#define DEFAULT_samplesFile 0
  int samplesFile = DEFAULT_samplesFile;

#define CHARARG_restart '^'
#define COMMENT_restart "Prefix of net save files. If 'none' then no restart."
#define TYPEVAL_restart string
#define TYPENUM_restart STRING
#define DEFAULT_restart "none"
  string restart = DEFAULT_restart;

#define CHARARG_maxTotSeqNum '='
#define COMMENT_maxTotSeqNum "DEPRECATED: Maximum number of sequences in \
training buffer"
#define TYPEVAL_maxTotSeqNum int
#define TYPENUM_maxTotSeqNum INT
#define DEFAULT_maxTotSeqNum 1000
  int maxTotSeqNum = DEFAULT_maxTotSeqNum;

#define CHARARG_randSeed '*'
#define COMMENT_randSeed "Random seed."
#define TYPEVAL_randSeed int
#define TYPENUM_randSeed INT
#define DEFAULT_randSeed 0
  int randSeed = DEFAULT_randSeed;

#define CHARARG_saveFreq ')'
#define COMMENT_saveFreq "Frequency of checkpoints of learner state. These \
checkpoints can be used to evaluate learners, but not yet to restart learning."
#define TYPEVAL_saveFreq int
#define TYPENUM_saveFreq INT
#define DEFAULT_saveFreq 1000000
  int saveFreq = DEFAULT_saveFreq;

///////////////////////////////////////////////////////////////////////////////
//SETTINGS PERTAINING TO ENVIRONMENT: NUMBER
///////////////////////////////////////////////////////////////////////////////
#define CHARARG_environment '0'
#define COMMENT_environment "Environment name, required is action/state \
space properties are hardcoded in smarties rather than read at runtime."
#define TYPEVAL_environment string
#define TYPENUM_environment STRING
#define DEFAULT_environment "Environment"
  string environment = DEFAULT_environment;

#define CHARARG_workersPerEnv '1'
#define COMMENT_workersPerEnv "Number of MPI ranks required to run each instance of the environment."
#define TYPEVAL_workersPerEnv int
#define TYPENUM_workersPerEnv INT
#define DEFAULT_workersPerEnv 1
  int workersPerEnv = DEFAULT_workersPerEnv;

#define CHARARG_rType '2'
#define COMMENT_rType "Reward type (can be defined by user in the environment)."
#define TYPEVAL_rType int
#define TYPENUM_rType INT
#define DEFAULT_rType 0
  int rType = DEFAULT_rType;

#define CHARARG_senses '3'
#define COMMENT_senses "Perceptions allowed to agent (can be defined by user \
in the environment)."
#define TYPEVAL_senses int
#define TYPENUM_senses INT
#define DEFAULT_senses 0
  int senses = DEFAULT_senses;

#define CHARARG_nStepPappSett '4'
#define COMMENT_nStepPappSett "Number of time steps per appSettings file to \
use. Must be a list of positive numbers separated by semicolons. Last number \
will be overwritten to 0; i.e. last appSettings will be used til termination."
#define TYPEVAL_nStepPappSett string
#define TYPENUM_nStepPappSett STRING
#define DEFAULT_nStepPappSett "0"
  string nStepPappSett = DEFAULT_nStepPappSett;

#define CHARARG_launchfile '7'
#define COMMENT_launchfile "Name of executable or launch script of user \
application. No arguments can go here. The file must be placed in the \
base run folder."
#define TYPEVAL_launchfile string
#define TYPENUM_launchfile STRING
#define DEFAULT_launchfile "launchSim.sh"
  string launchfile = DEFAULT_launchfile;

#define CHARARG_appSettings '8'
#define COMMENT_appSettings "Name of file containing the command line arguments for user's application."
#define TYPEVAL_appSettings string
#define TYPENUM_appSettings STRING
#define DEFAULT_appSettings ""
  string appSettings = DEFAULT_appSettings;

#define CHARARG_setupFolder '9'
#define COMMENT_setupFolder "The contents of this folder are copied over into the folder where the simulation is run. It can contain additional files needed to set up the simulation such as settings files, configuration files..."
#define TYPEVAL_setupFolder string
#define TYPENUM_setupFolder STRING
#define DEFAULT_setupFolder ""
string setupFolder = DEFAULT_setupFolder;

#define READOPT(NAME)  { CHARARG_ ## NAME, #NAME, TYPENUM_ ## NAME, \
  COMMENT_ ## NAME, &NAME, (TYPEVAL_ ## NAME) DEFAULT_ ## NAME }

///////////////////////////////////////////////////////////////////////////////
//SETTINGS THAT ARE NOT READ FROM FILE
///////////////////////////////////////////////////////////////////////////////
  MPI_Comm mastersComm;
  int world_rank = 0;
  int world_size = 0;
  int workers_rank = 0;
  int workers_size = 0;
  int learner_rank = 0;
  int learner_size = 0;
  // number of workers (usually per master)
  int nWorkers = 1;
  //number of agents that:
  // in case of worker: # of agents that are contained in an environment
  // in case of master: nWorkers * # are contained in an environment
  int nAgents = -1;
  // whether Recurrent network (figured out in main)
  bool bRecurrent = false;
  // number of quantities defining the policy, depends on env and algorithm
  int policyVecDim = -1;
  //random number generators (one per thread)
  //std::mt19937* gen;
  int threadSafety = -1;
  std::vector<std::mt19937> generators;

  void check()
  {
    bRecurrent = nnType=="LSTM" || nnType=="RNN" || nnType == "MGU" || nnType == "GRU";

    if(bSampleSequences && maxTotSeqNum<batchSize)
    die("Increase memory buffer size or decrease batchsize, or switch to sampling by transitions.");
    if(bTrain == false && restart == "none") {
     cout<<"Did not specify path for restart files, assumed current dir."<<endl;
     restart = ".";
    }
    if(appendedObs<0)  die("appendedObs<0");
    if(targetDelay<0)  die("targetDelay<0");
    if(splitLayers<0)  die("splitLayers<0");
    if(totNumSteps<0)  die("totNumSteps<0");
    if(sockPrefix<0)   die("sockPrefix<0");
    if(obsPerStep<0)   die("obsPerStep<0");
    if(learnrate>.1)   die("learnrate>.1");
    if(learnrate<0)    die("learnrate<0");
    if(explNoise<0)    die("explNoise<0");
    if(epsAnneal<0)    die("epsAnneal<0");
    if(batchSize<0)    die("batchSize<0");
    if(nnLambda<0)     die("nnLambda<0");
    if(nThreads<1)     die("nThreads<1");
    if(nMasters<1)     die("nMasters<1");
    if(gamma<0)        die("gamma<0");
    if(gamma>1)        die("gamma>1");
    if(nnl1<0)         die("nnl1<0");
    if(nnl2<0)         die("nnl2<0");
    if(nnl3<0)         die("nnl3<0");
    if(nnl4<0)         die("nnl4<0");
    if(nnl5<0)         die("nnl5<0");
    if(epsAnneal>0.0001 || epsAnneal<0) {
      warn("epsAnneal should be tiny. It will be set to 5e-7 for this run.");
      epsAnneal = 5e-7;
    }
  }

  vector<ArgParser::OptionStruct> initializeOpts ()
  { //  //{ CHARARG_, "", TYPENUM_, COMMENT_, &, (TYPEVAL_) DEFAULT_ },
    //AVERT YOUR EYES!

    return vector<ArgParser::OptionStruct> ({
      // LEARNER ARGS: MUST contain all 17 mentioned above (more if modified)
      READOPT(learner), READOPT(bTrain), READOPT(clipImpWeight),
      READOPT(targetDelay), READOPT(explNoise), READOPT(ERoldSeqFilter),
      READOPT(gamma), READOPT(klDivConstraint), READOPT(lambda),
      READOPT(minTotObsNum), READOPT(maxTotObsNum), READOPT(obsPerStep),
      READOPT(penalTol), READOPT(epsAnneal), READOPT(bSampleSequences),
      READOPT(bSharedPol), READOPT(totNumSteps),

      // NETWORK ARGS: MUST contain all 15 mentioned above (more if modified)
      READOPT(nnl1), READOPT(nnl2), READOPT(nnl3), READOPT(nnl4),
      READOPT(nnl5), READOPT(nnl6), READOPT(batchSize), READOPT(appendedObs),
      READOPT(nnPdrop), READOPT(nnOutputFunc), READOPT(nnFunc),
      READOPT(learnrate), READOPT(nnType), READOPT(outWeightsPrefac),
      READOPT(nnLambda), READOPT(splitLayers), READOPT(nnBPTTseq),

      // SMARTIES ARGS: MUST contain all 10 mentioned above (more if modified)
      READOPT(nThreads), READOPT(nMasters), READOPT(isServer), READOPT(ppn),
      READOPT(sockPrefix), READOPT(samplesFile), READOPT(restart),
      READOPT(maxTotSeqNum), READOPT(randSeed), READOPT(saveFreq),

      // ENVIRONMENT ARGS: MUST contain all 7 mentioned above (more if modified)
      READOPT(environment), READOPT(workersPerEnv), READOPT(rType),
      READOPT(senses), READOPT(nStepPappSett),
      READOPT(launchfile), READOPT(appSettings), READOPT(setupFolder)
    });
  }

  void initRandomSeed()
  {
    if(randSeed<=0) {
      struct timeval clock;
      gettimeofday(&clock, NULL);
      const long MAXINT = std::numeric_limits<int>::max();
      randSeed = abs(clock.tv_usec % MAXINT);
      MPI_Bcast(&randSeed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    sockPrefix = randSeed + world_rank;

    generators.reserve(omp_get_max_threads());
    generators.push_back(mt19937(sockPrefix));
    for(int i=1; i<omp_get_max_threads(); i++) {
      const Uint seed = generators[0]();
      generators.push_back(mt19937(seed));
    }
  }

  void finalizeSeeds()
  {
    const int currsize = generators.size();
    if(currsize < nThreads + nAgents) {
      generators.reserve(nThreads+nAgents);
      for(int i=currsize; i<nThreads+nAgents; i++) {
        const Uint seed = generators[0]();
        generators.push_back(mt19937(seed));
      }
    }
  }

  vector<int> readNetSettingsSize()
  {
    vector<int> ret;
    //if(nnl1<1) die("Add at least one hidden layer.\n");
    if(nnl1>0) {
      ret.push_back(nnl1);
      if (nnl2>0) {
        ret.push_back(nnl2);
        if (nnl3>0) {
          ret.push_back(nnl3);
          if (nnl4>0) {
            ret.push_back(nnl4);
            if (nnl5>0) {
              ret.push_back(nnl5);
              if (nnl6>0) {
                ret.push_back(nnl6);
              }
            }
          }
        }
      }
    }
    return ret;
  }
};
