//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "AlgoFactory.h"

#include "PPO.h"
#include "DPG.h"
#include "DQN.h"
#include "NAF.h"
#include "ACER.h"
#include "RACER.h"
#include "CMALearner.h"
#include "Learner_pytorch.h"

#include <fstream>
#include <sstream>
#include <unistd.h>

namespace smarties
{

inline static void printLogfile(std::ostringstream&o, std::string fn, int rank)
{
  if(rank != 0) return;
  std::ofstream fout(fn.c_str(), std::ios::app);
  fout << o.str() << std::endl;
  fout.flush();
  fout.close();
}

inline std::ifstream findSettingsFile(DistributionInfo& D, const Uint ID)
{
  char currDirectory[512];
  getcwd(currDirectory, 512);
  chdir(D.initial_runDir);

  // TODO: allow user to set name?
  std::ifstream ret;
  char settingsName[256];
  sprintf(settingsName, "settings_%02lu.json", ID);
  ret.open(settingsName, std::ifstream::in);
  // if found a json for this learner specifically, then read it
  if( ret.is_open() ) {
    chdir(currDirectory);
    return ret;
  }

  // else return the default settings name for all settings files:
  ret.open("settings.json", std::ifstream::in);
  chdir(currDirectory);
  return ret;
}

std::unique_ptr<Learner> createLearner(
  const Uint learnerID, MDPdescriptor& MDP, DistributionInfo& distrib
)
{
  char lName[256];
  sprintf(lName, "agent_%02lu", learnerID);
  if(distrib.world_rank == 0)
    printf("Creating learning algorithm #%02lu\n", learnerID);

  const ActionInfo aInfo = ActionInfo(MDP);
  Settings settings;
  std::ifstream ifs = findSettingsFile(distrib, learnerID);
  settings.initializeOpts(ifs, distrib);

  std::unique_ptr<Learner> ret;
  std::ostringstream o;
  o << MDP.dimState << " ";

  if (settings.learner == "PYTORCH")
  {
    ret = std::make_unique<Learner_pytorch>(MDP, settings, distrib);
  }
  else
  if (settings.learner == "RACER")
  {
    if(MDP.bDiscreteActions) {
      using RACER_discrete = RACER<Discrete_advantage, Discrete_policy, Uint>;
      MDP.policyVecDim = RACER_discrete::getnDimPolicy(&aInfo);
      o << MDP.maxActionLabel << " " << MDP.policyVecDim;
      printLogfile(o, "problem_size.log", distrib.world_rank);
      ret = std::make_unique<RACER_discrete>(MDP, settings, distrib);
    } else {
      //using RACER_continuous = RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>;
      using RACER_continuous = RACER<Param_advantage,Gaussian_policy,Rvec>;
      MDP.policyVecDim = RACER_continuous::getnDimPolicy(&aInfo);
      o << MDP.dimAction << " " << MDP.policyVecDim;
      printLogfile(o, "problem_size.log", distrib.world_rank);
      ret = std::make_unique<RACER_continuous>(MDP, settings, distrib);
    }
  }
  else
  if (settings.learner == "VRACER")
  {
    if(MDP.bDiscreteActions) {
      // V-RACER with discrete actions makes little sense: we revert to RACER
      warn("V-RACER makes little sense with discrete action-spaces. Code will "
        "override user and add parameterization for action advantage.");
      using RACER_discrete = RACER<Discrete_advantage, Discrete_policy, Uint>;
      MDP.policyVecDim = RACER_discrete::getnDimPolicy(&aInfo);
      o << MDP.maxActionLabel << " " << MDP.policyVecDim;
      printLogfile(o, "problem_size.log", distrib.world_rank);
      ret = std::make_unique<RACER_discrete>(MDP, settings, distrib);
    } else {
      //using RACER_continuous = VRACER<Gaussian_mixture<NEXPERTS>, Rvec>;
      using RACER_continuous = RACER<Zero_advantage, Gaussian_policy, Rvec>;
      MDP.policyVecDim = RACER_continuous::getnDimPolicy(&aInfo);
      o << MDP.dimAction << " " << MDP.policyVecDim;
      printLogfile(o, "problem_size.log", distrib.world_rank);
      ret = std::make_unique<RACER_continuous>(MDP, settings, distrib);
    }
  }
  else
  if (settings.learner == "DDPG" || settings.learner == "DPG")
  {
    MDP.policyVecDim = 2*MDP.dimAction;
    // non-NPER DPG is unstable with annealed network learn rate
    // because critic network must adapt quickly
    if(settings.clipImpWeight<=0) settings.epsAnneal = 0;
    o << MDP.dimAction << " " << MDP.policyVecDim;
    printLogfile(o, "problem_size.log", distrib.world_rank);
    ret = std::make_unique<DPG>(MDP, settings, distrib);
  }
  else
  if (settings.learner == "GAE" || settings.learner == "PPO")
  {
    if(MDP.bDiscreteActions) {
      using PPO_discrete = PPO<Discrete_policy, Uint>;
      MDP.policyVecDim = PPO_discrete::getnDimPolicy(&aInfo);
      o << MDP.maxActionLabel << " " << MDP.policyVecDim;
      printLogfile(o, "problem_size.log", distrib.world_rank);
      ret = std::make_unique<PPO_discrete>(MDP, settings, distrib);
    } else {
      using PPO_continuous = PPO<Gaussian_policy, Rvec>;
      MDP.policyVecDim = PPO_continuous::getnDimPolicy(&aInfo);
      o << MDP.dimAction << " " << MDP.policyVecDim;
      printLogfile(o, "problem_size.log", distrib.world_rank);
      ret = std::make_unique<PPO_continuous>(MDP, settings, distrib);
    }
  }
  else
  if (settings.learner == "ACER")
  {
    settings.bSampleSequences = true;
    if(MDP.bDiscreteActions)
      die("implemented ACER supports only continuous-action problems");
    MDP.policyVecDim = ACER::getnDimPolicy(&aInfo);
    o << MDP.dimAction << " " << MDP.policyVecDim;
    printLogfile(o, "problem_size.log", distrib.world_rank);
    ret = std::make_unique<ACER>(MDP, settings, distrib);
  }
  else
  if(settings.learner=="NFQ" || settings.learner=="DQN")
  {
    if(not MDP.bDiscreteActions)
      die("DQN supports only discrete-action problems");
    o << MDP.maxActionLabel << " " << MDP.maxActionLabel;
    printLogfile(o, "problem_size.log", distrib.world_rank);
    MDP.policyVecDim = MDP.maxActionLabel;
    ret = std::make_unique<DQN>(MDP, settings, distrib);
  }
  else
  if (settings.learner == "NA" || settings.learner == "NAF")
  {
    MDP.policyVecDim = 2*MDP.dimAction;
    assert(not MDP.bDiscreteActions);
    o << MDP.dimAction << " " << MDP.policyVecDim;
    printLogfile(o, "problem_size.log", distrib.world_rank);
    ret = std::make_unique<NAF>(MDP, settings, distrib);
  }
  else
  if (settings.learner == "CMA")
  {
    //if(settings.ESpopSize<2)
    //  die("Must be coupled with CMA. Set ESpopSize>1");
    //if( settings.nWorkers % settings.learner_size )
    //  die("nWorkers must be multiple of learner ranks");
    //if( settings.ESpopSize % settings.learner_size )
    //  die("CMA pop size must be multiple of learners");
    //if( settings.ESpopSize % settings.nWorkers )
    //  die("CMA pop size must be multiple of nWorkers");

    if(MDP.bDiscreteActions) {
      using CMA_discrete = CMALearner<Uint>;
      MDP.policyVecDim = CMA_discrete::getnDimPolicy(&aInfo);
      o << MDP.maxActionLabel << " " << MDP.policyVecDim;
      printLogfile(o, "problem_size.log", distrib.world_rank);
      ret = std::make_unique<CMA_discrete>(MDP, settings, distrib);
    } else {
      using CMA_continuous = CMALearner<Rvec>;
      MDP.policyVecDim = CMA_continuous::getnDimPolicy(&aInfo);
      if(settings.explNoise > 0) MDP.policyVecDim += MDP.dimAction;
      o << MDP.dimAction << " " << MDP.policyVecDim;
      printLogfile(o, "problem_size.log", distrib.world_rank);
      ret = std::make_unique<CMA_continuous>(MDP, settings, distrib);
    }
  }
  /*
  */
  else die("Learning algorithm not recognized");

  ret->setLearnerName(std::string(lName)+"_", learnerID);

  return ret;
}

}
