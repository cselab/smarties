//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "../StateAction.h"
#include "../Network/Builder.h"
#include "NAF.h"

NAF::NAF(Environment*const _env, Settings & _set) :
Learner_offPolicy(_env, _set)
{
  trainInfo = new TrainData("NAF", _set);
  F.push_back(new Approximator("value", _set, input, data));
  Builder build_pol = F[0]->buildFromSettings(_set, aInfo.dim);
  F[0]->initializeNetwork(build_pol);
  test();
}

void NAF::select(Agent& agent)
{
  const Real annealedVar = explNoise + (bTrain ? annealingFactor() : 0);
  Sequence* const traj = data->inProgress[agent.ID];
  data->add_state(agent);
  F[0]->prepare_agent(traj, agent);

  if( agent.Status < TERM_COMM )
  {
    //Compute policy and value on most recent element of the sequence. If RNN
    // recurrent connection from last call from same agent will be reused
    Rvec output = F[0]->forward_agent<CUR>(traj, agent);
    const Quadratic_advantage advantage = prepare_advantage(output);
    Rvec pol = advantage.getMean();
    pol.resize(policyVecDim, annealedVar);
    const auto act=Gaussian_policy::sample(&generators[nThreads+agent.ID], pol);
    agent.act(aInfo.getScaled(act));
    data->add_action(agent, pol);
  } else
    data->terminate_seq(agent);
}

void NAF::TrainBySequences(const Uint seq, const Uint thrID) const
{
  Sequence* const traj = data->Set[seq];
  const Uint ndata = traj->tuples.size();
  F[0]->prepare_seq(traj, thrID);
  if(thrID==0) profiler->stop_start("FWD");

  for (Uint k=0; k<ndata-1; k++) { //state in k=[0:N-2]
    const bool terminal = k+2==ndata && traj->ended;
    const Rvec output = F[0]->forward<CUR>(traj, k, thrID);
    const Real Vsold = output[net_indices[0]];
    const Rvec act = aInfo.getInvScaled(traj->tuples[k]->a);
    const Quadratic_advantage adv_sold = prepare_advantage(output);
    const Real Qsold = Vsold + adv_sold.computeAdvantage(act);

    Real Vsnew = traj->tuples[k+1]->r;
    if ( not terminal ) {
      const Rvec target = F[0]->forward<TGT>(traj, k+1, thrID);
      Vsnew += gamma*target[net_indices[0]];
    }
    const Real error = Vsnew - Qsold;
    Rvec gradient(F[0]->nOutputs());
    gradient[net_indices[0]] = error;
    adv_sold.grad(act, error, gradient);

    traj->setMseDklImpw(k, error*error, 0, 1);
    trainInfo->log(Qsold, error, thrID);
    F[0]->backward(gradient, k, thrID);
  }

  if(thrID==0)  profiler->stop_start("BCK");
  F[0]->gradient(thrID);
}

void NAF::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
  Sequence* const traj = data->Set[seq];
  F[0]->prepare_one(traj, samp, thrID);
  if(thrID==0) profiler->stop_start("FWD");

  const Rvec output = F[0]->forward<CUR>(traj, samp, thrID);
  const bool terminal = samp+2 == traj->tuples.size() && traj->ended;
  const Real Vsold = output[net_indices[0]];
  //unbounded action:
  const Rvec act = aInfo.getInvScaled(traj->tuples[samp]->a);
  const Quadratic_advantage adv_sold = prepare_advantage(output);
  const Real Qsold = Vsold + adv_sold.computeAdvantage(act);

  Real Vsnew = traj->tuples[samp+1]->r;
  if (not terminal) {
    const Rvec target = F[0]->forward<TGT>(traj, samp+1, thrID);
    Vsnew += gamma*target[net_indices[0]];
  }
  const Real error = Vsnew - Qsold;
  Rvec gradient(F[0]->nOutputs());
  gradient[net_indices[0]] = error;
  adv_sold.grad(act, error, gradient);

  trainInfo->log(Qsold, error, thrID);
  traj->setMseDklImpw(samp, error*error, 0, 1);
  if(thrID==0)  profiler->stop_start("BCK");
  F[0]->backward(gradient, samp, thrID);
  F[0]->gradient(thrID);
}
