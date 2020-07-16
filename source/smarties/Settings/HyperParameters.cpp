//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "HyperParameters.h"

#include "ExecutionInfo.h"
#include "../Utils/Warnings.h"

#include "../../extern/json.hpp"
#include <cassert>
#include <fstream>
#include <unistd.h>

namespace smarties
{

std::string HyperParameters::printArgComments()
{
  using json = nlohmann::json;
  json j;
  j["learner"] =
    "Chosen learning algorithm. One of: "
    "'RACER', 'VRACER', 'PPO', 'DPG', 'ACER', 'NAF', 'DQN', 'CMA', 'PYTORCH'.";
  //
  j["explNoise"] =
    "Noise added to policy. For discrete policies it may be the probability of "
    "picking a random action (detail depend on learning algo), for continuous "
    "policies it is the (initial) standard deviation.";
  //
  j["gamma"] = "Discount factor.";
  //
  j["lambda"] = "Lambda for off-policy return-based estimators.";
  //
  j["obsPerStep"] =
    "Ratio of observed *transitions* to gradient steps. E.g. 0.1 "
    "means that for every observation learner does 10 gradient steps.";
  //
  j["clipImpWeight"] =
    "Clipping range for off-policy importance weights. Corresponds to: C in "
    "ReF-ER's Rule 1, epsilon in PPO's pol objective, c in ACER's truncation.";
  //
  j["penalTol"] =
    "Tolerance used for adaptive off-policy penalization methods. "
    "Currently corresponds only to D in ReF-ER's Rule 2.";
  //
  j["klDivConstraint"] =
    "Constraint on max KL div. Corresponds to: d_targ in PPO's penalization, "
    "delta in ACER's truncation and bias correction.";
  //
  j["targetDelay"] =
    "Copy delay for Target Nets (TNs). If 0, TNs are disabled. If 'val'>1: "
    "every 'val' grad steps network's W copied onto TN (like DQN). If 'val'<1: "
    "every grad step TN updated by exp. averaging with rate 'val' (like DPG).";
  //
  j["epsAnneal"] =
    "Annealing rate for various learning-algorithm-dependent behaviors.";
  //
  j["ERoldSeqFilter"] =
    "Filter algorithm to remove old episodes from memory buffer. Accepts: "
    "'oldest', 'farpolfrac', 'maxkldiv', 'minerror', or 'default'. Default "
    "means 'oldest' for ER and 'farpolfrac' for ReFER.";
  //
  j["dataSamplingAlgo"] = "Algorithm for sampling the Replay Buffer.";
  //
  j["minTotObsNum"] =
    "Min number of transitions in training buffer before training starts. "
    "If minTotObsNum=0, is set equal to maxTotObsNum i.e. fill RM before "
    "training starts.";
  //
  j["maxTotObsNum"] = "Max number of transitions in training buffer.";
  //
  j["saveFreq"] = "Frequency of checkpoints for learner state.";
  //
  j["encoderLayerSizes"] =
    "Sizes of non-convolutional encoder layers (LSTM/RNN/FFNN). E.g. '64 64'.";
  //
  j["nnLayerSizes"] =
    "Sizes of non-convolutional layers (LSTM/RNN/FFNN). E.g. '64 64'.";
  //
  j["batchSize"] = "Network training batch size.";
  //
  j["nnOutputFunc"] = "Activation function for output layers.";
  //
  j["nnFunc"] =
    "Activation function for non-output layers (which is almost always "
    "linear) which are built from settings. ('Relu', 'Tanh', 'Sigm', 'PRelu', "
    "'softSign', 'softPlus', ...)";
  //
  j["learnrate"] = "Learning rate.";
  //
  j["ESpopSize"] =
    "Population size for ES algorithm. If unset, or set to <2, we use Adam.";
  //
  j["nnType"] =
    "Type of non-output layers read from settings. (RNN, LSTM, everything else "
    "maps to FFNN). Conv2D layers need to be built in environment directly.";
  //
  j["outWeightsPrefac"] = "Output weights initialization factor (will be multiplied by default fan-in factor). Picking 1 leads to treating output layers with normal Xavier initialization.";
  //
  j["nnLambda"] = "Penalization factor for network weights. It will be \
  multiplied by learn rate: w -= eta * nnLambda * w . L1 decay option in Bund.h";
  //
  j["nnBPTTseq"] = "Number of previous steps considered by RNN.";
  //
  return j.dump(4);
}

void HyperParameters::initializeOpts(std::ifstream & inputStream,
                                     ExecutionInfo & distrib)
{
  using json = nlohmann::json;
  json j;
  // if we actually have a good ifstream, read the json, else will use defaults
  if( inputStream.is_open() ) inputStream >> j;

  // LEARNING ALGORITHM
  if(!j["learner"].empty())                   learner = j["learner"];
  if(!j["ERoldSeqFilter"].empty())     ERoldSeqFilter = j["ERoldSeqFilter"];
  if(!j["dataSamplingAlgo"].empty()) dataSamplingAlgo = j["dataSamplingAlgo"];
  if(!j["returnsEstimator"].empty()) returnsEstimator = j["returnsEstimator"];

  if(!j["explNoise"].empty())               explNoise = j["explNoise"];
  if(!j["gamma"].empty())                       gamma = j["gamma"];
  if(!j["lambda"].empty())                     lambda = j["lambda"];
  if(!j["obsPerStep"].empty())             obsPerStep = j["obsPerStep"];
  if(!j["clipImpWeight"].empty())       clipImpWeight = j["clipImpWeight"];
  if(!j["penalTol"].empty())                 penalTol = j["penalTol"];
  if(!j["klDivConstraint"].empty())   klDivConstraint = j["klDivConstraint"];
  if(!j["targetDelay"].empty())           targetDelay = j["targetDelay"];
  if(!j["epsAnneal"].empty())               epsAnneal = j["epsAnneal"];

  if(!j["minTotObsNum"].empty())         minTotObsNum = j["minTotObsNum"];
  if(!j["maxTotObsNum"].empty())         maxTotObsNum = j["maxTotObsNum"];
  if(!j["saveFreq"].empty())                 saveFreq = j["saveFreq"];

  // NETWORK APPROXIMATORS
  if(!j["encoderLayerSizes"].empty()) {
    const std::vector<int> tmp = j["encoderLayerSizes"];
    encoderLayerSizes.resize(tmp.size());
    std::copy(tmp.begin(), tmp.end(), encoderLayerSizes.begin());
  }
  if(!j["nnLayerSizes"].empty()) {
    const std::vector<int> tmp = j["nnLayerSizes"];
    nnLayerSizes.resize(tmp.size());
    std::copy(tmp.begin(), tmp.end(), nnLayerSizes.begin());
  }

  if(!j["batchSize"].empty())               batchSize = j["batchSize"];
  if(!j["ESpopSize"].empty())               ESpopSize = j["ESpopSize"];
  if(!j["nnBPTTseq"].empty())               nnBPTTseq = j["nnBPTTseq"];
  if(!j["nnLambda"].empty())                 nnLambda = j["nnLambda"];
  if(!j["learnrate"].empty())               learnrate = j["learnrate"];
  if(!j["outWeightsPrefac"].empty()) outWeightsPrefac = j["outWeightsPrefac"];
  if(!j["nnOutputFunc"].empty())         nnOutputFunc = j["nnOutputFunc"];
  if(!j["nnFunc"].empty())                     nnFunc = j["nnFunc"];
  if(!j["nnType"].empty())                     nnType = j["nnType"];

  // split read workloads among processes:
  defineDistributedLearning(distrib);
  check();
}

void HyperParameters::defineDistributedLearning(ExecutionInfo& distrib)
{
  const MPI_Comm& learnersComm = distrib.learners_train_comm;
  //const MPI_Comm& gatheringComm = distrib.master_workers_comm;
  const Uint nLearners = learnersComm==MPI_COMM_NULL? 1
                         : MPICommSize(learnersComm);
  const Real nL = nLearners;
  // each learner computes a fraction of the batch:
  if(batchSize > 1) {
    batchSize = std::ceil(batchSize / nL) * nL;
    batchSize_local = batchSize / nLearners;
  } else batchSize_local = batchSize;

  if(minTotObsNum <= 0) minTotObsNum = maxTotObsNum;
  minTotObsNum = std::min(minTotObsNum, maxTotObsNum); 
  minTotObsNum = std::ceil(minTotObsNum / nL) * nL;
  minTotObsNum_local = minTotObsNum / nLearners;
  // each learner processes a fraction of the entire dataset:
  maxTotObsNum = std::ceil(maxTotObsNum / nL) * nL;
  maxTotObsNum_local = maxTotObsNum / nLearners;

  // each worker collects a fraction of the initial memory buffer:
  const Real nOwnEnvs = distrib.nOwnedEnvironments;
  obsPerStep_local = nOwnEnvs * obsPerStep / distrib.nEnvironments;
  //obsPerStep_local = obsPerStep;
  if(batchSize_local <= 0) die(" ");
  if(maxTotObsNum_local <= 0) die(" ");
}

void HyperParameters::check()
{
  bRecurrent= nnType=="LSTM" || nnType=="RNN" || nnType=="MGU" || nnType=="GRU";
  if(targetDelay<0) die("targetDelay<0");
  if(obsPerStep<0)  die("obsPerStep<0");
  if(learnrate>1)   die("learnrate>1");
  if(learnrate<0)   die("learnrate<0");
  if(explNoise<0)   die("explNoise<0");
  if(epsAnneal<0)   die("epsAnneal<0");
  if(batchSize<=0)  die("batchSize<0");
  if(nnLambda<0)    die("nnLambda<0");
  if(gamma<0)       die("gamma<0");
  if(gamma>1)       die("gamma>1");
  if(epsAnneal>0.0001 || epsAnneal<0) {
    warn("epsAnneal should be tiny. It will be set to 5e-7 for this run.");
    epsAnneal = 5e-7;
  }
}

}
