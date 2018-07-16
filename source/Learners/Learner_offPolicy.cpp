//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learner_offPolicy.h"

Learner_offPolicy::Learner_offPolicy(Environment*const _env, Settings & _s) :
Learner(_env,_s), obsPerStep_orig(_s.obsPerStep), nObsPerTraining(
_s.minTotObsNum>_s.batchSize? _s.minTotObsNum : _s.maxTotObsNum) {
  if(not bSampleSequences && nObsPerTraining < batchSize)
    die("Parameter minTotObsNum is too low for given problem");
}

bool Learner_offPolicy::readyForTrain() const
{
  //const Uint nTransitions = data->readNTransitions();
  //if(data->nSequences>=data->adapt_TotSeqNum && nTransitions<nData_b4Train())
  //  die("I do not have enough data for training. Change hyperparameters");
  //const Real nReq = std::sqrt(data->readAvgSeqLen()*16)*batchSize;
  const bool ready = bTrain && data->readNData() >= nObsPerTraining;

  if(not ready && bTrain && learn_rank==0)
  {
    lock_guard<mutex> lock(buffer_mutex);
    const int currPerc = data->readNData() * 100. / (Real) nObsPerTraining;
    if(currPerc>=percData+5) {
      percData = currPerc;
      printf("\rCollected %d%% of data required to begin training. ", percData);
      fflush(0); //otherwise no show on some platforms
    }
  }
  return ready;
}

bool Learner_offPolicy::lockQueue() const
{
  // lockQueue tells scheduler that has stopped receiving states from workers
  // whether should start communication again.
  // for off policy learning, there is a ratio between gradient steps
  // and observed transitions to be kept (approximatively) constant

  //if there is not enough data for training, need more data
  if( not readyForTrain() ) return false;

  //const Real _nData = (Real)data->readNConcluded() - nData_b4Startup;
  const Real _nData = data->readNSeen() - nData_b4Startup;
  const Real dataCounter = _nData - (Real)nData_last;
  const Real stepCounter =  nStep - (Real)nStep_last;
  // Lock the queue if we have !added to the training set! more observations
  // than (grad_step * obsPerStep) or if the update is ready.
  // The distinction between "added to set" and "observed" allows removing
  // some load inbalance, with only has marginal effects on algorithms.
  // Load imb. is reduced by minimizing pauses in either data or grad stepping.
  const bool tooMuchData = dataCounter > stepCounter*obsPerStep;
  return tooMuchData;
}

void Learner_offPolicy::spawnTrainTasks_par()
{
  // it should be impossible to get here before starting batch update was ready
  if(updateComplete || updateToApply) die("undefined behavior");

  if( not readyForTrain() ) {
    warn("spawnTrainTasks_par called with not enough data, wait next call");
    // This can happen if data pruning algorithm is allowed to delete a lot of
    // data from the mem buffer, which could cause training to pause
    return; // Do not prepare an update
  }

  if(bSampleSequences && data->readNSeq() < batchSize)
    die("Parameter minTotObsNum is too low for given problem");

  profiler->stop_start("SAMP");
  debugL("Sample the replay memory and compute the gradients");
  vector<Uint> samp_seq = vector<Uint>(batchSize, -1);
  vector<Uint> samp_obs = vector<Uint>(batchSize, -1);
  if(bSampleSequences) data->sampleSequences(samp_seq);
  else data->sampleTransitions(samp_seq, samp_obs);

  profiler->stop_start("SLP"); // so we see inactive time during parallel loop
  #pragma omp parallel for schedule(dynamic) num_threads(nThreads)
  for (Uint i=0; i<batchSize; i++)
  {
    Uint seq = samp_seq[i], obs = samp_obs[i];
    const int thrID = omp_get_thread_num();
    //printf("Thread %d done %u %u %f\n",thrID,seq,obs,data->Set[seq]->offPolicImpW[obs]); fflush(0);
    if(bSampleSequences)
    {
      obs = data->Set[seq]->ndata()-1;
      TrainBySequences(seq, thrID);
      #pragma omp atomic
      nAddedGradients += data->Set[seq]->ndata();
    }
    else
    {
      Train(seq, obs, thrID);
      #pragma omp atomic
      nAddedGradients++;
    }

    input->gradient(thrID);
    data->Set[seq]->setSampled(obs);
    if(thrID==0) profiler->stop_start("SLP");
  }

  updateComplete = true;
}

bool Learner_offPolicy::bNeedSequentialTrain() {return false;}
void Learner_offPolicy::spawnTrainTasks_seq() { }

void Learner_offPolicy::applyGradient()
{
  const auto currStep = nStep+1; // base class will advance this with this func
  if(updateToApply)
  {
    debugL("Prune the Replay Memory for old/stale episodes, advance counters");
    //put here because this is called after workers finished gathering new data
    profiler->stop_start("PRNE");
    //shift data / gradient counters to maintain grad stepping to sample
    // collection ratio prescirbed by obsPerStep
    const Real stepCounter = currStep - (Real)nStep_last;
    assert(std::fabs(stepCounter-1) < nnEPS);
    nData_last += stepCounter*obsPerStep;
    nStep_last = currStep;

    if(CmaxPol>0) // assume ReF-ER
    {
      CmaxRet = 1 + annealRate(CmaxPol, currStep, epsAnneal);
      if(CmaxRet<=1) die("Either run lasted too long or epsAnneal is wrong.");
      data->prune(REFER_FILTER, CmaxRet);
      Real fracOffPol = data->nOffPol / (Real) data->readNData();

      if (learn_size > 1) {
        vector<Real> partial_data {(Real)data->nOffPol,(Real)data->readNData()};
        // use result from prev AllReduce to update rewards (before new reduce).
        // Assumption is that the number of off Pol trajectories does not change
        // much each step. Especially because here we update the off pol W only
        // if an obs is actually sampled. Therefore at most this fraction
        // is wrong by batchSize / nTransitions ( ~ 0 )
        // In exchange we skip an mpi implicit barrier point.
        const bool skipped = reductor.sync(partial_data);
        fracOffPol = partial_data[0] / partial_data[1];
        if(skipped and partial_data[0]>nnEPS)
          die("If skipping it must be 1st step, with nothing far policy");
      }

      if(fracOffPol>ReFtol) beta = (1-learnR)*beta; // iter converges to 0
      else beta = learnR +(1-learnR)*beta; //fixed point iter converge to 1

      if( beta <= 10*learnR && currStep % 1000 == 0)
      warn("beta too low. Lower lrate, pick bounded nnfunc, or incr net size.");
    }
    else
    {
      data->prune(MEMBUF_FILTER_ALGO);
    }
  }
  else
  {
    if( not readyForTrain() ) die("undefined behavior");
    warn("Pruning at prev grad step removed too much data and training was paused: shift training counters");
    // Prune at prev grad step removed too much data and training was paused.
    // ApplyGradient was surely called by Scheduler after workers finished
    // gathering new data enabling training to continue ( after workers.join() )
    // Therefore we should shift these counters to restart gradient stepping:
    nData_b4Startup = data->readNConcluded();
    nData_last = 0;
  }

  if( readyForTrain() )
  {
    debugL("Compute state/rewards stats from the replay memory");
    // placed here because this occurs after workers.join() so we have new data
    profiler->stop_start("PRE");
    if(currStep%1000==0) { // update state mean/std with net's learning rate
      const Real WS = annealRate(learnR, currStep, epsAnneal);
      data->updateRewardsStats(currStep, 1, WS*(OFFPOL_ADAPT_STSCALE>0));
    }
  }
  else
  {
    warn("Pruning removed too much data from buffer: will have to wait one scheduler loop before training can continue");
  }

  Learner::applyGradient();
}

void Learner_offPolicy::initializeLearner()
{
  if ( not readyForTrain() || nStep>0 ) die("undefined behavior");

  // shift counters after initial data is gathered
  nData_b4Startup = data->readNConcluded();
  nData_last = 0;

  debugL("Compute state/rewards stats from the replay memory");
  profiler->stop_start("PRE");
  data->updateRewardsStats(nStep, 1, 1);
  if( learn_rank == 0 )
    cout<<"Initial reward std "<<1/data->invstd_reward<<endl;

  Learner::initializeLearner();
}
