//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_HyperParameters_h
#define smarties_HyperParameters_h

#include "Definitions.h"
#include "../Utils/MPIUtilities.h"

#include <random>
#include <mutex>

namespace smarties
{

struct ExecutionInfo;

struct HyperParameters
{
  const Uint dimS, dimA;

  HyperParameters(const Uint nS, const Uint nA) : dimS(nS), dimA(nA) {}

  void check();
  static std::string printArgComments();
  void initializeOpts(std::ifstream & , ExecutionInfo & );
  void defineDistributedLearning(ExecutionInfo &);

  //////////////////////////////////////////////////////////////////////////////
  //SETTINGS PERTAINING TO LEARNING ALGORITHM
  //////////////////////////////////////////////////////////////////////////////
  std::string learner = "VRACER";
  std::string ERoldSeqFilter = "oldest";
  std::string dataSamplingAlgo = "uniform";
  std::string returnsEstimator = "default";

  Real explNoise = std::sqrt(0.2);
  Real gamma = 0.995;
  Real lambda = 1;
  Real obsPerStep = 1;
  Real clipImpWeight = std::sqrt(dimA / 2.0);
  Real penalTol = 0.1;
  Real klDivConstraint = 0.01;
  Real targetDelay = 0;
  Real epsAnneal = 5e-7;

  Uint minTotObsNum = 0;
  Uint maxTotObsNum = std::pow(2, 14) * std::sqrt(dimA + dimS);
  Uint saveFreq = 200000;

  //////////////////////////////////////////////////////////////////////////////
  //SETTINGS PERTAINING TO NETWORK
  //////////////////////////////////////////////////////////////////////////////

  std::vector<Uint> encoderLayerSizes = { 0 };
  std::vector<Uint> nnLayerSizes = { 128, 128 };

  Uint batchSize = 256;
  Uint ESpopSize = 1;
  Uint nnBPTTseq = 16;

  Real nnLambda = std::numeric_limits<float>::epsilon();
  Real learnrate = 1e-4;
  Real outWeightsPrefac = 1e-3;

  std::string nnOutputFunc = "Linear";
  std::string nnFunc = "Tanh";
  std::string nnType = "FFNN";

  //////////////////////////////////////////////////////////////////////////////
  //SETTINGS THAT ARE NOT READ FROM FILE
  //////////////////////////////////////////////////////////////////////////////
  // rank-local data-acquisition goals:
  Uint batchSize_local = 0;
  Real obsPerStep_local = 0;
  Uint minTotObsNum_local = 0;
  Uint maxTotObsNum_local = 0;
  // whether Recurrent network (figured out in main)
  bool bRecurrent = false;
  // whether sampling minibatch of episodes or of timesteps, determined by algo
  bool bSampleEpisodes = false;
};

} // end namespace smarties
#endif // smarties_Settings_h
