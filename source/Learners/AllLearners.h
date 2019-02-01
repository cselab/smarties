//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "DQN.h"
#include "ACER.h"
#include "NAF.h"
#include "DPG.h"
#include "RETPG.h"
#include "VRACER.h"
#include "RACER.h"
#include "CMALearner.h"
#include "PPO.h"

inline void print(std::ostringstream& o, std::string fname, int rank)
{
  if(rank != 0) return;
  ofstream fout(fname.c_str(), ios::app);
  fout << o.str() << endl;
  fout.flush();
  fout.close();
}

inline Learner* createLearner(Environment*const env, Settings&settings)
{
  Learner * ret = nullptr;
  std::ostringstream o;
  o << env->sI.dim << " ";

  if(settings.learner=="NFQ" || settings.learner=="DQN") {
    assert(env->aI.discrete);
    o << env->aI.maxLabel << " " << env->aI.maxLabel;
    print(o, "problem_size.log", settings.world_rank);
    env->aI.policyVecDim = env->aI.maxLabel;
    ret = new DQN(env, settings);
  }
  else if (settings.learner == "ACER") {
    settings.bSampleSequences = true;
    assert(env->aI.discrete == false);
    env->aI.policyVecDim = ACER::getnDimPolicy(&env->aI);
    o << env->aI.dim << " " << env->aI.policyVecDim;
    print(o, "problem_size.log", settings.world_rank);
    ret = new ACER(env, settings);
  }
  else
  if (settings.learner == "NA" || settings.learner == "NAF") {
    settings.bSampleSequences = false;
    env->aI.policyVecDim = 2*env->aI.dim;
    assert(not env->aI.discrete);
    o << env->aI.dim << " " << env->aI.policyVecDim;
    print(o, "problem_size.log", settings.world_rank);
    ret = new NAF(env, settings);
  }
  else
  if (settings.learner == "DP" || settings.learner == "DPG") {
    settings.bSampleSequences = false;
    env->aI.policyVecDim = 2*env->aI.dim;
    // non-NPER DPG is unstable with annealed network learn rate
    // because critic network must adapt quickly
    if(settings.clipImpWeight<=0) settings.epsAnneal = 0;
    o << env->aI.dim << " " << env->aI.policyVecDim;
    print(o, "problem_size.log", settings.world_rank);
    ret = new DPG(env, settings);
  }
  else
  if (settings.learner == "RETPG") {
    settings.bSampleSequences = false;
    env->aI.policyVecDim = 2*env->aI.dim;
    // non-NPER DPG is unstable with annealed network learn rate
    // because critic network must adapt quickly
    if(settings.clipImpWeight<=0) settings.epsAnneal = 0;
    o << env->aI.dim << " " << env->aI.policyVecDim;
    print(o, "problem_size.log", settings.world_rank);
    ret = new RETPG(env, settings);
  }
  else
  if (settings.learner == "RACER") {
    if(env->aI.discrete) {
      using RACER_discrete = RACER<Discrete_advantage, Discrete_policy, Uint>;
      env->aI.policyVecDim = RACER_discrete::getnDimPolicy(&env->aI);
      o << env->aI.maxLabel << " " << env->aI.policyVecDim;
      print(o, "problem_size.log", settings.world_rank);
      ret = new RACER_discrete(env, settings);
    } else {
      //using RACER_continuous = RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>;
      using RACER_continuous = RACER<Param_advantage,Gaussian_policy,Rvec>;
      env->aI.policyVecDim = RACER_continuous::getnDimPolicy(&env->aI);
      o << env->aI.dim << " " << env->aI.policyVecDim;
      print(o, "problem_size.log", settings.world_rank);
      ret = new RACER_continuous(env, settings);
    }
  }
  else
  if (settings.learner == "CMA") {
    settings.batchSize_loc = settings.batchSize;
    if(settings.ESpopSize<2)
      die("Must be coupled with CMA. Set ESpopSize>1");
    if( settings.nWorkers % settings.learner_size )
      die("nWorkers must be multiple of learner ranks");
    if( settings.ESpopSize % settings.learner_size )
      die("CMA pop size must be multiple of learners");
    if( settings.ESpopSize % settings.nWorkers )
      die("CMA pop size must be multiple of nWorkers");

    if(env->aI.discrete) {
      using CMA_discrete = CMALearner<Uint>;
      env->aI.policyVecDim = CMA_discrete::getnDimPolicy(&env->aI);
      o << env->aI.maxLabel << " " << env->aI.policyVecDim;
      print(o, "problem_size.log", settings.world_rank);
      ret = new CMA_discrete(env, settings);
    } else {
      using CMA_continuous = CMALearner<Rvec>;
      env->aI.policyVecDim = CMA_continuous::getnDimPolicy(&env->aI);
      if(settings.explNoise > 0) env->aI.policyVecDim += env->aI.dim;
      o << env->aI.dim << " " << env->aI.policyVecDim;
      print(o, "problem_size.log", settings.world_rank);
      ret = new CMA_continuous(env, settings);
    }
  }
  else
  if (settings.learner == "GAE" || settings.learner == "PPO") {
    settings.bSampleSequences = false;
    if(env->aI.discrete) {
      using PPO_discrete = PPO<Discrete_policy, Uint>;
      env->aI.policyVecDim = PPO_discrete::getnDimPolicy(&env->aI);
      o << env->aI.maxLabel << " " << env->aI.policyVecDim;
      print(o, "problem_size.log", settings.world_rank);
      ret = new PPO_discrete(env, settings);
    } else {
      using PPO_continuous = PPO<Gaussian_policy, Rvec>;
      env->aI.policyVecDim = PPO_continuous::getnDimPolicy(&env->aI);
      o << env->aI.dim << " " << env->aI.policyVecDim;
      print(o, "problem_size.log", settings.world_rank);
      ret = new PPO_continuous(env, settings);
    }
  }
  else
  if (settings.learner == "VRACER") {
    if(env->aI.discrete) {
      using RACER_discrete = VRACER<Discrete_policy, Uint>;
      env->aI.policyVecDim = RACER_discrete::getnDimPolicy(&env->aI);
      o << env->aI.maxLabel << " " << env->aI.policyVecDim;
      print(o, "problem_size.log", settings.world_rank);
      ret = new RACER_discrete(env, settings);
    } else {
      //using RACER_continuous = VRACER<Gaussian_mixture<NEXPERTS>, Rvec>;
      using RACER_continuous = VRACER<Gaussian_policy, Rvec>;
      env->aI.policyVecDim = RACER_continuous::getnDimPolicy(&env->aI);
      o << env->aI.dim << " " << env->aI.policyVecDim;
      print(o, "problem_size.log", settings.world_rank);
      ret = new RACER_continuous(env, settings);
    }
  }
  //else die("Learning algorithm not recognized\n");

  assert(ret not_eq nullptr);
  return ret;
}
