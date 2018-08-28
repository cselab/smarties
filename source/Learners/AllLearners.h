//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "DQN.h"
#include "NAF.h"
#include "DPG.h"
#include "RETPG.h"
#include "RACER.h"
#include "VRACER.h"
#include "ACER.h"
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
    settings.policyVecDim = env->aI.maxLabel;
    ret = new DQN(env, settings);
  }
  else if (settings.learner == "RACER") {
    if(env->aI.discrete) {
      using RACER_discrete = RACER<Discrete_advantage, Discrete_policy, Uint>;
      settings.policyVecDim = RACER_discrete::getnDimPolicy(&env->aI);
      o << env->aI.maxLabel << " " << settings.policyVecDim;
      print(o, "problem_size.log", settings.world_rank);
      ret = new RACER_discrete(env, settings);
    } else {
      using RACER_continuous = RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>;
      //typedef RACER_cont RACER_continuous;
      settings.policyVecDim = RACER_continuous::getnDimPolicy(&env->aI);
      o << env->aI.dim << " " << settings.policyVecDim;
      print(o, "problem_size.log", settings.world_rank);
      ret = new RACER_continuous(env, settings);
    }
  }
  else if (settings.learner == "VRACER") {
    if(env->aI.discrete) {
      using RACER_discrete = VRACER<Discrete_policy, Uint>;
      settings.policyVecDim = RACER_discrete::getnDimPolicy(&env->aI);
      o << env->aI.maxLabel << " " << settings.policyVecDim;
      print(o, "problem_size.log", settings.world_rank);
      ret = new RACER_discrete(env, settings);
    } else {
      using RACER_continuous = VRACER<Gaussian_mixture<NEXPERTS>, Rvec>;
      settings.policyVecDim = RACER_continuous::getnDimPolicy(&env->aI);
      o << env->aI.dim << " " << settings.policyVecDim;
      print(o, "problem_size.log", settings.world_rank);
      ret = new RACER_continuous(env, settings);
    }
  }
  else if (settings.learner == "ACER") {
    settings.bSampleSequences = true;
    assert(env->aI.discrete == false);
    settings.policyVecDim = ACER::getnDimPolicy(&env->aI);
    o << env->aI.dim << " " << settings.policyVecDim;
    print(o, "problem_size.log", settings.world_rank);
    ret = new ACER(env, settings);
  }
  else if (settings.learner == "NA" || settings.learner == "NAF") {
    settings.bSampleSequences = false;
    settings.policyVecDim = 2*env->aI.dim;
    assert(not env->aI.discrete);
    o << env->aI.dim << " " << settings.policyVecDim;
    print(o, "problem_size.log", settings.world_rank);
    ret = new NAF(env, settings);
  }
  else if (settings.learner == "DP" || settings.learner == "DPG") {
    settings.bSampleSequences = false;
    settings.policyVecDim = 2*env->aI.dim;
    // non-NPER DPG is unstable with annealed network learn rate
    // because critic network must adapt quickly
    if(settings.clipImpWeight<=0) settings.epsAnneal = 0;
    o << env->aI.dim << " " << settings.policyVecDim;
    print(o, "problem_size.log", settings.world_rank);
    ret = new DPG(env, settings);
  }
  else if (settings.learner == "RETPG") {
    settings.bSampleSequences = false;
    settings.policyVecDim = 2*env->aI.dim;
    // non-NPER DPG is unstable with annealed network learn rate
    // because critic network must adapt quickly
    if(settings.clipImpWeight<=0) settings.epsAnneal = 0;
    o << env->aI.dim << " " << settings.policyVecDim;
    print(o, "problem_size.log", settings.world_rank);
    ret = new RETPG(env, settings);
  }
  else if (settings.learner == "GAE" || settings.learner == "PPO") {
    settings.bSampleSequences = false;
    if(env->aI.discrete) {
      using PPO_discrete = PPO<Discrete_policy, Uint>;
      settings.policyVecDim = PPO_discrete::getnDimPolicy(&env->aI);
      o << env->aI.maxLabel << " " << settings.policyVecDim;
      print(o, "problem_size.log", settings.world_rank);
      ret = new PPO_discrete(env, settings);
    } else {
      using PPO_continuous = PPO<Gaussian_policy, Rvec>;
      settings.policyVecDim = PPO_continuous::getnDimPolicy(&env->aI);
      o << env->aI.dim << " " << settings.policyVecDim;
      print(o, "problem_size.log", settings.world_rank);
      ret = new PPO_continuous(env, settings);
    }
  } else die("Learning algorithm not recognized\n");

  env->aI.policyVecDim = settings.policyVecDim;
  assert(ret not_eq nullptr);
  return ret;
}
