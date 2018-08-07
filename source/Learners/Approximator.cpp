//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Approximator.h"
#include "../Network/Builder.h"

void Aggregator::prepare(const RELAY SET, const Uint thrID) const
{
  usage[thrID] = SET;
}

void Aggregator::prepare(const Uint N, const Sequence*const traj,
  const Uint samp, const Uint thrID, const RELAY SET) const
{
  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  const Uint nTotal = nRecurr + N;
  first_sample[thrID] = samp - nRecurr;
  inputs[thrID].clear(); //make sure we only have empty vectors
  inputs[thrID].resize(nTotal, Rvec());
  usage[thrID] = SET;
}

void Aggregator::prepare_seq(const Sequence*const traj, const Uint thrID, const RELAY SET) const
{
  const Uint nSValues =  traj->tuples.size() - traj->ended;
  first_sample[thrID] = 0;
  inputs[thrID].clear(); //make sure we only have empty vectors
  inputs[thrID].resize(nSValues, Rvec());
  usage[thrID] = SET;
}

void Aggregator::prepare_one(const Sequence*const traj, const Uint samp,
    const Uint thrID, const RELAY SET) const
{
  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  // might need to predict the value of next state if samp not terminal state
  const Uint nTotal = nRecurr + 2;
  first_sample[thrID] = samp - nRecurr;
  inputs[thrID].clear(); //make sure we only have empty vectors
  inputs[thrID].resize(nTotal, Rvec());
  usage[thrID] = SET;
}

void Aggregator::set(const Rvec vec,const Uint samp,const Uint thrID) const
{
  usage[thrID] = VEC;
  const int ind = (int)samp - first_sample[thrID];
  assert(first_sample[thrID] <= (int)samp);
  assert(ind >= 0 && (int) inputs[thrID].size() > ind);
  inputs[thrID][ind] = vec;
}

Rvec Aggregator::get(const Sequence*const traj, const Uint samp,
    const Uint thrID, const PARAMS USEW) const
{
  Rvec ret;
  if(usage[thrID] == VEC) {
    assert(first_sample[thrID] >= 0);
    const int ind = (int)samp - first_sample[thrID];
    assert(first_sample[thrID] <= (int)samp);
    assert(ind >= 0 && (int) inputs[thrID].size() > ind);
    assert(inputs[thrID][ind].size());
    ret = inputs[thrID][ind];
  } else if (usage[thrID] == ACT) {
    assert(aI.dim == nOuts);
    ret = aI.getInvScaled(traj->tuples[samp]->a);
  } else {
    ret = approx->forward(traj, samp, thrID, USEW, USEW);
    assert(ret.size() >= nOuts); // in DPG we now also output parametric stdev
    ret.resize(nOuts);           // vector, therefore ... this workaround
  }
  for(Uint j=0; j<nOuts; j++) assert(!std::isnan(ret[j])&&!std::isinf(ret[j]));
  return scale(ret);
}

Builder Approximator::buildFromSettings(Settings&sett, const vector<Uint>nouts)
{
  Builder build(sett);
  Uint nInputs = input->nOutputs() + (relay==nullptr ? 0 : relay->nOutputs());
  build.stackSimple( nInputs, nouts );
  return build;
}

Builder Approximator::buildFromSettings(Settings& _s, const Uint n_outputs) {
  Builder build(_s);
  Uint nInputs = input->nOutputs() + (relay==nullptr ? 0 : relay->nOutputs());
  build.stackSimple( nInputs, {n_outputs} );
  return build;
}

void Approximator::allocMorePerThread(const Uint nAlloc)
{
  assert(nAlloc > 0 && extraAlloc == 0);
  extraAlloc = nAlloc;
  assert(opt not_eq nullptr && net not_eq nullptr);
  series.resize(nThreads*(1+nAlloc));

  for (Uint j=1; j<=nAlloc; j++)
    #pragma omp parallel for schedule(static, 1) num_threads(nThreads)
      for (Uint i = j*nThreads; i<(1+j)*nThreads; i++)
        #pragma omp critical
          series[i].reserve(MAX_SEQ_LEN);
}

void Approximator::initializeNetwork(Builder& build, Real cutGradFactor)
{
  net = build.build();
  opt = build.opt;
  assert(opt not_eq nullptr && net not_eq nullptr);

  for (Uint i=0; i<nAgents; i++) agent_series[i].reserve(2);

  #pragma omp parallel for schedule(static, 1) num_threads(nThreads)
  for (Uint i=0; i<nThreads; i++) // numa aware allocation
   #pragma omp critical
   {
     series[i].reserve(MAX_SEQ_LEN);
     series_tgt[i].reserve(MAX_SEQ_LEN);
   }

  if(relay not_eq nullptr) {
    vector<int> relayInputID;
    for(Uint i=1; i<net->layers.size(); i++) //assume layer 0 is passed to input
      if(net->layers[i]->bInput) relayInputID.push_back(i);

    if(relayInputID.size() > 1) { die("should not be possible");
    } else if (relayInputID.size() == 1) {
      relayInp = relayInputID[0];
      if(net->layers[relayInp]->nOutputs() != relay->nOutputs()) die("crap");
    } else relayInp = 0;
  }

  if(not net->layers[0]->bInput) die("should not be possible");
  // skip backprop to input vector or to input features if `blockInpGrad`
  if ( input->net == nullptr or blockInpGrad ) {
    Uint layBckPrpInp = 1, nInps = input->nOutputs();
    // make sure that we are computing relay gradient
    if(relayInp>0) { //then lay 0 is input, 1 is relay, 2 is joining
      layBckPrpInp = 3;
      if(not net->layers[1]->bInput) die("should not be possible"); //relay
      if(net->layers[2]->bInput) die("should not be possible"); //joining
    }
    if(relay==nullptr) {
      if(net->layers[layBckPrpInp]->spanCompInpGrads!=nInps)
        die("should not be possible");
    } else
      if(net->layers[layBckPrpInp]->spanCompInpGrads!=nInps+relay->nOutputs())
        die("should not be possible");

    net->layers[layBckPrpInp]->spanCompInpGrads -= nInps;
    net->layers[layBckPrpInp]->startCompInpGrads = nInps;
  }

  #ifdef __CHECK_DIFF //check gradients with finite differences
    net->checkGrads();
  #endif
  gradStats = new StatsTracker(net->getnOutputs(), settings, cutGradFactor);
}

void Approximator::prepare(const Uint N, const Sequence*const traj,
  const Uint samp, const Uint thrID, const Uint nSamples) const
{
  if(error_placements[thrID] > 0) die("");
  assert(nSamples<=1+extraAlloc && nSamples>0);
  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  const Uint nTotal = nRecurr + N;
  input->prepare(nTotal, samp - nRecurr, thrID);

  for(Uint k=0; k<nSamples; k++)
    net->prepForBackProp(series[thrID + k*nThreads], nTotal);

  if(series_tgt.size()>thrID)
    net->prepForFwdProp(series_tgt[thrID], nTotal);

  first_sample[thrID] = samp - nRecurr;
}

void Approximator::prepare_seq(const Sequence*const traj, const Uint thrID,
  const Uint nSamples) const
{
  if(error_placements[thrID] > 0) die("");
  assert(nSamples<=1+extraAlloc && nSamples>0);
  const Uint nSValues =  traj->tuples.size() - traj->ended;
  input->prepare(nSValues, 0, thrID);

  for(Uint k=0; k<nSamples; k++)
    net->prepForBackProp(series[thrID + k*nThreads], nSValues);

  if(series_tgt.size()>thrID)
    net->prepForFwdProp(series_tgt[thrID], nSValues);

  first_sample[thrID] = 0;
}

void Approximator::prepare_one(const Sequence*const traj, const Uint samp,
    const Uint thrID, const Uint nSamples) const
{
  if(error_placements[thrID] > 0) die("");
  assert(nSamples<=1+extraAlloc && nSamples>0);
  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  // might need to predict the value of next state if samp not terminal state
  const Uint nTotal = nRecurr + 2;

  input->prepare(nTotal, samp - nRecurr, thrID);
  for(Uint k=0; k<nSamples; k++)
    net->prepForBackProp(series[thrID + k*nThreads], nTotal);

  net->prepForFwdProp(series_tgt[thrID], nTotal);

  first_sample[thrID] = samp - nRecurr;
}

Rvec Approximator::forward(const Sequence* const traj, const Uint samp,
  const Uint thrID, const PARAMS USE_WEIGHTS, const PARAMS USE_ACT,
  const Uint iSample, const int overwrite) const
{
  if(iSample) assert(USE_ACT == CUR && iSample<=extraAlloc);
  if(thrID>=nThreads) return forward_agent(traj, thrID-nThreads, USE_WEIGHTS);
  const Uint netID = thrID + iSample*nThreads;
  const vector<Activation*>& act=USE_ACT==CUR? series[netID] :series_tgt[thrID];
  const vector<Activation*>& act_cur = series[thrID];
  const int ind = mapTime2Ind(samp, thrID);

  //if already computed just give answer
  if(act[ind]->written == true && not overwrite)
    return act[ind]->getOutput();

  // write previous outputs if needed (note: will spawn nested function calls)
  if(ind>0 && act_cur[ind-1]->written not_eq true)
    this->forward(traj, samp-1, thrID);

  const Rvec inp = getInput(traj, samp, thrID, USE_WEIGHTS);
  //cout <<"Input : "<< print(inp) << endl; fflush(0);
  return getOutput(inp, ind, act[ind], thrID, USE_WEIGHTS);
}

Rvec Approximator::relay_backprop(const Rvec err,
  const Uint samp, const Uint thrID, const PARAMS USEW) const
{
  if(relay == nullptr || relayInp < 0) die("improperly set up the relay");
  const vector<Activation*>& act = series_tgt[thrID];
  const int ind = mapTime2Ind(samp, thrID), nInp = input->nOutputs();
  assert(act[ind]->written == true && relay not_eq nullptr);
  const Parameters*const W = USEW==CUR? net->weights : net->tgt_weights;
  const Rvec ret = net->inpBackProp(err, act[ind], W, relayInp);
  for(Uint j=0; j<ret.size(); j++)
    assert(!std::isnan(ret[j]) && !std::isinf(ret[j]));
  //if(!thrID) {
  //  const auto pret = Rvec(&ret[nInp], &ret[nInp+relay->nOutputs()]);
  //  const auto inp = act[ind]->getInput();
  //  const auto pinp = Rvec(&inp[nInp], &inp[nInp+relay->nOutputs()]);
  //  cout <<"G:"<<print(pret)<< " Inp:"<<print(pinp)<<endl;
  //}
  if(relayInp>0) return relay->scale(ret);
  else return relay->scale(Rvec(&ret[nInp], &ret[nInp+relay->nOutputs()]));
}

void Approximator::prepare_agent(const Sequence*const traj, const Agent&agent) const
{
  //This is called by a std::thread and uses separate workspace from omp threads
  //We use a fake thread id to avoid code duplication in encapsulator class
  const Uint fakeThrID = nThreads + agent.ID, stepid = traj->ndata();
  // learner->select always only gets one new state, so we assume that it needs
  // to run one (or more) forward net at time t, so here also compute recurrency
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT,stepid) : 0;
  const vector<Activation*>& act = agent_series[agent.ID];
  net->prepForFwdProp(agent_series[agent.ID], nRecurr+1);
  input->prepare(nRecurr+1, stepid-nRecurr, fakeThrID);
  // if using relays, ask for previous actions, to be used for recurrencies
  // why? because the past is the past.
  if(relay not_eq nullptr) relay->prepare(ACT, fakeThrID);
  assert(act.size() >= nRecurr+1);

  //Advance recurr net with 0 initialized activations for nRecurr steps
  for(Uint i=0; i<nRecurr; i++) {
    const Rvec inp = getInput(traj, stepid-nRecurr+i, fakeThrID, CUR);
    net->predict(inp, i>0? act[i-1] : nullptr, act[i], net->weights);
  }
}

Rvec Approximator::forward_agent(const Sequence* const traj,
  const Uint agentID, const PARAMS USEW) const
{
  // assume we already computed recurrencies
  const vector<Activation*>& act = agent_series[agentID];
  const Uint fakeThrID = nThreads + agentID, stepid = traj->ndata();
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT,stepid) : 0;
  if(act[nRecurr]->written) return act[nRecurr]->getOutput();
  const Parameters* const W = USEW==CUR? net->weights : net->tgt_weights;
  const Rvec inp = getInput(traj, stepid, fakeThrID, USEW);
  return net->predict(inp, nRecurr? act[nRecurr-1] : nullptr, act[nRecurr], W);
}

Rvec Approximator::getOutput(const Rvec inp, const int ind,
  Activation*const act, const Uint thrID, const PARAMS USEW) const
{
  //hardcoded to use time series predicted with cur weights for recurrencies:
  const vector<Activation*>& act_cur = series[thrID];
  const Activation*const recur = ind? act_cur[ind-1] : nullptr;
  const Parameters* const W = USEW==CUR? net->weights : net->tgt_weights;
  const Rvec ret = net->predict(inp, recur, act, W);
  //if(!thrID) cout<<"net fwd with inp:"<<print(inp)<<" out:"<<print(ret)<<endl;
  act->written = true;
  return ret;
}

Rvec Approximator::getInput(const Sequence*const traj, const Uint samp,
  const Uint thrID, const PARAMS USE_WEIGHTS) const
{
  Rvec inp = input->forward(traj, samp, thrID);
  if(relay not_eq nullptr) {
    const Rvec addedinp = relay->get(traj, samp, thrID, USE_WEIGHTS);
    assert(addedinp.size());
    inp.insert(inp.end(), addedinp.begin(), addedinp.end());
    //if(!thrID) cout << "relay "<<print(addedinp) << endl;
  }
  assert(inp.size() == net->getnInputs());
  return inp;
}

void Approximator::backward(Rvec error, const Sequence*const traj, const Uint t,
  const Uint thrID, const Uint iSample) const
{
  const Uint netID = thrID + iSample*nThreads;
  #ifdef PRIORITIZED_ER
   const Real anneal = std::min( (Real)1, opt->nStep * opt->epsAnneal);
   assert( anneal >= 0 );
   const float beta = 0.5 + 0.5 * anneal, P0 = traj->priorityImpW[t];
   // if samples never seen by optimizer the samples have high priority
   // this matches one of last lines of MemoryBuffer::updateImportanceWeights()
   const auto P = P0<=0 ? data->maxPriorityImpW : P0;
   const Real PERW = std::pow(data->minPriorityImpW / P, beta);
   for(Uint i=0; i<error.size(); i++) error[i] *= PERW;
  #endif
  gradStats->track_vector(error, thrID);
  gradStats->clip_vector(error);
  const int ind = mapTime2Ind(t, thrID);
  const vector<Activation*>& act = series[netID];
  assert(act[ind]->written == true && iSample <= extraAlloc);
  //ind+1 because we use c-style for loops in other places: TODO:netID
  error_placements[thrID] = std::max(ind+1, error_placements[thrID]);
  act[ind]->addOutputDelta(error);
}

void Approximator::prepareUpdate(const Uint batchSize)
{
  for(Uint i=0; i<nThreads; i++) if(error_placements[i]>0) die("");

  if(nAddedGradients==0) die("No-gradient update. Revise hyperparameters.");
  if(nAddedGradients>batchSize) die("weird");

  opt->prepare_update(batchSize, net->Vgrad);
  reducedGradients = 1;
  nAddedGradients = 0;
  if(mpisize<=1) applyUpdate();
}

void Approximator::applyUpdate()
{
  if(reducedGradients == 0) return;

  opt->apply_update();
  net->updateTransposed();
  reducedGradients = 0;
}

void Approximator::gradient(const Uint thrID) const
{
  if(error_placements[thrID]<=0) die("");

  #pragma omp atomic
  nAddedGradients++;

  for(Uint j = 0; j<=extraAlloc; j++)
  {
    const Uint netID = thrID + j*nThreads;
    const vector<Activation*>& act = series[netID];
    const int last_error = error_placements[thrID];

    for (int i=0; i<last_error; i++) assert(act[i]->written == true);

    net->backProp(act, last_error, net->Vgrad[thrID]);
    //for(int i=0;i<last_error&&!thrID;i++)cout<<i<<" inpG:"<<print(act[i]->getInputGradient(0))<<endl;

    if(input->net == nullptr || blockInpGrad) continue;

    for(int i=0; i<last_error; i++) {
      Rvec inputG = act[i]->getInputGradient(0);
      inputG.resize(input->nOutputs());
      input->backward(inputG, first_sample[thrID] +i, thrID);
    }
  }
  error_placements[thrID] = -1; //to stop additional backprops
}

void Approximator::getHeaders(ostringstream& buff) const
{
  buff << std::left << std::setfill(' ') <<"| " << std::setw(6) << name;
  if(opt->tgtUpdateAlpha > 0) buff << "| dTgt ";
}

void Approximator::getMetrics(ostringstream& buff) const
{
  real2SS(buff, net->weights->compute_weight_norm(), 7, 1);
  if(opt->tgtUpdateAlpha > 0)
    real2SS(buff, net->weights->compute_weight_dist(net->tgt_weights), 6, 1);
}
