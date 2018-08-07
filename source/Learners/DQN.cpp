//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "DQN.h"
#include "../Math/Utils.h"
#include "../Network/Builder.h"

DQN::DQN(Environment*const _env, Settings& _set) :
Learner_offPolicy(_env, _set)
{
  trainInfo = new TrainData("DQN", _set);
  F.push_back(new Approximator("Q", _set, input, data));
  Builder build_pol = F[0]->buildFromSettings(_set, env->aI.maxLabel);
  F[0]->initializeNetwork(build_pol);
}

void DQN::select(Agent& agent)
{
  const Real anneal = annealingFactor();
  const Real annealedEps = bTrain ? anneal + (1-anneal)*explNoise : explNoise;
  Sequence* const traj = data->inProgress[agent.ID];
  data->add_state(agent);

  if( agent.Status < TERM_COMM )
  {
    //Compute policy and value on most recent element of the sequence. If RNN
    // recurrent connection from last call from same agent will be reused
    F[0]->prepare_agent(traj, agent);
    Rvec output = F[0]->forward_agent<CUR>(traj, agent);

    uniform_real_distribution<Real> dis(0.,1.);
    if(dis(generators[nThreads+agent.ID]) < annealedEps)
      agent.act(env->aI.maxLabel*dis(generators[nThreads+agent.ID]));
    else agent.act(maxInd(output));

    Rvec mu(policyVecDim, annealedEps/env->aI.maxLabel);
    mu[maxInd(output)] += 1-annealedEps;

    data->add_action(agent, mu);
  } else
    data->terminate_seq(agent);
}

void DQN::TrainBySequences(const Uint seq, const Uint thrID) const
{
  Sequence* const traj = data->Set[seq];
  const Uint ndata = traj->tuples.size();
  F[0]->prepare_seq(traj, thrID);
  if(thrID==0) profiler->stop_start("FWD");

  for (Uint k=0; k<ndata-1; k++) { //state in k=[0:N-2]
    const bool terminal = k+2==ndata && traj->ended;
    const Rvec Qs = F[0]->forward<CUR>(traj, k, thrID);
    const Uint action = aInfo.actionToLabel(traj->tuples[k]->a);

    Real Vsnew = traj->tuples[k+1]->r;
    if ( not terminal ) {
      const Rvec Qhats = F[0]->forward<CUR>(traj, k+1, thrID);
      const Rvec Qtildes = F[0]->forward<TGT>(traj, k+1, thrID);
      Vsnew += gamma * Qtildes[maxInd(Qhats)];
    }
    const Real error = Vsnew - Qs[action];

    Rvec gradient(F[0]->nOutputs());
    gradient[action] = error;
    traj->setMseDklImpw(k, error*error, 0, 1);
    trainInfo->log(Qs[action], error, thrID);
    F[0]->backward(gradient, traj, k, thrID);
  }

  if(thrID==0)  profiler->stop_start("BCK");
  F[0]->gradient(thrID);
}

void DQN::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
  Sequence* const traj = data->Set[seq];
  F[0]->prepare_one(traj, samp, thrID);
  if(thrID==0) profiler->stop_start("FWD");

  const Rvec Qs = F[0]->forward<CUR>(traj, samp, thrID);
  const bool terminal = samp+2 == traj->tuples.size() && traj->ended;
  const Uint act = aInfo.actionToLabel(traj->tuples[samp]->a);

  Real Vsnew = traj->tuples[samp+1]->r;
  if (not terminal) {
    // find best action for sNew with moving wghts, evaluate it with tgt wgths:
    // Double Q Learning ( http://arxiv.org/abs/1509.06461 )
    const Rvec Qhats = F[0]->forward<CUR>(traj, samp+1, thrID);
    const Rvec Qtildes = F[0]->forward<TGT>(traj, samp+1, thrID);
    Vsnew += gamma * Qtildes[maxInd(Qhats)];
  }
  const Real error = Vsnew - Qs[act];
  Rvec gradient(F[0]->nOutputs(), 0);
  gradient[act] = error;

  traj->setMseDklImpw(samp, error*error, 0, 1);
  trainInfo->log(Qs[act], error, thrID);
  if(thrID==0)  profiler->stop_start("BCK");
  F[0]->backward(gradient, traj, samp, thrID);
  F[0]->gradient(thrID);
}
