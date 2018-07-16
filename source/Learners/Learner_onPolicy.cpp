//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learner_onPolicy.h"

Learner_onPolicy::Learner_onPolicy(Environment*const _env, Settings&_s): Learner(_env, _s), nHorizon(_s.maxTotObsNum),
nEpochs(_s.batchSize/_s.obsPerStep) {
  cout << "nHorizon:"<<nHorizon<<endl;
  cout << "nEpochs:"<<nEpochs<<endl;
}
// nHorizon is number of obs in buffer during training
// steps per epoch = nEpochs * nHorizon / batchSize
// obs per step = nHorizon / (steps per epoch)
// this leads to formula used to compute nEpochs


// unlockQueue tells scheduler that has stopped receiving states from workers
// whether should start communication again.
// for on policy learning, when enough data is collected from workers
// gradient stepping starts and collection is paused
// when training is concluded collection restarts
bool Learner_onPolicy::lockQueue() const
{
  return bTrain && data->readNData() >= nHorizon + cntKept;
}

bool Learner_onPolicy::bNeedSequentialTrain() {return true;}
void Learner_onPolicy::spawnTrainTasks_seq()
{
  if(updateComplete || updateToApply) die("undefined behavior");
  if( data->readNData() < nHorizon ) die("undefined behavior");
  if( not bTrain ) return;

  debugL("sampling update from (nearly) on-pol data");
  vector<Uint> samp_seq(batchSize, -1), samp_obs(batchSize, -1);
  data->sampleTransitions(samp_seq, samp_obs);

  profiler->stop_start("SLP"); // so we see inactive time during parallel loop
  #pragma omp parallel for schedule(dynamic) num_threads(nThreads)
  for (Uint i=0; i<batchSize; i++)
  {
    const int thrID = omp_get_thread_num();
    const Uint seq = samp_seq[i], obs = samp_obs[i];
    Train(seq, obs, thrID);
    input->gradient(thrID);
    data->Set[seq]->setSampled(obs);
    if(thrID==0) profiler->stop_start("SLP");
  }
  updateComplete = true;
}

void Learner_onPolicy::spawnTrainTasks_par() { }

void Learner_onPolicy::prepareGradient()
{
  if(not updateComplete || updateToApply) die("undefined behavior");

  Learner::prepareGradient();

  debugL("shift counters of epochs over the stored data");
  cntBatch += batchSize;
  if(cntBatch >= nHorizon) {
    cntBatch = 0;
    cntEpoch++;
  }

  if(cntEpoch >= nEpochs) {
    debugL("finished epochs, compute state/rew stats, clear buffer to gather new onpol samples");
    const Real annlLR = annealRate(learnR, nStep, epsAnneal);
    data->updateRewardsStats(nStep, annlLR, annlLR);
    cntKept = data->clearOffPol(CmaxPol, 0.05);
    //reset batch learning counters
    cntEpoch = 0;
    cntBatch = 0;
  }
}

void Learner_onPolicy::initializeLearner()
{
  if ( nStep>0 ) die("undefined behavior");

  debugL("Compute state/rewards stats from the replay memory");
  profiler->stop_start("PRE");
  data->updateRewardsStats(nStep, 1, 1);
  if( learn_rank == 0 )
    cout<<"Initial reward std "<<1/data->invstd_reward<<endl;
}
