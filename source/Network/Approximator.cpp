//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Approximator.h"
#include "Aggregator.h"
#include "Builder.h"

Approximator::Approximator(const string _name, Settings&S, Encapsulator*const E,
  MemoryBuffer* const data_ptr, const Aggregator* const r) :
settings(S), name(_name), input(E), data(data_ptr), relay(r) { }

Approximator::~Approximator()
{
  _dispose_object(relay);
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

void Approximator::allocMorePerThread(const Uint nAlloc) {
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

void Approximator::initializeNetwork(Builder& build) {
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

  if(not net->layers[0]->bInput) {
    warn("Network has no input.");
  }
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
      if(net->layers.size() < layBckPrpInp)
      if(net->layers[layBckPrpInp]->spanCompInpGrads!=nInps)
        die("should not be possible");
    } else
      if(net->layers[layBckPrpInp]->spanCompInpGrads!=nInps+relay->nOutputs())
        die("should not be possible");

    if(net->layers.size() < layBckPrpInp) {
      net->layers[layBckPrpInp]->spanCompInpGrads -= nInps;
      net->layers[layBckPrpInp]->startCompInpGrads = nInps;
    }
  }

  #ifdef __CHECK_DIFF //check gradients with finite differences
    net->checkGrads();
  #endif
  gradStats = new StatsTracker(net->getnOutputs(), settings);
}

void Approximator::prepare_seq(Sequence*const traj, const Uint thrID,
  const Uint wghtID) const {
  if(error_placements[thrID] > 0) die("");
  input->prepare(traj, traj->tuples.size(), 0, thrID);

  for(Uint k=0; k < 1+extraAlloc; k++)
    net->prepForBackProp(series[thrID + k*nThreads], traj->tuples.size());

  if(series_tgt.size()>thrID)
    net->prepForFwdProp(series_tgt[thrID], traj->tuples.size());

  first_sample[thrID] = 0;
  thread_Wind[thrID] = wghtID;
  thread_seq[thrID] = traj;
}

void Approximator::prepare_one(Sequence*const traj, const Uint samp,
    const Uint thrID, const Uint wghtID) const {
  if(error_placements[thrID] > 0) die("");
  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  // might need to predict the value of next state if samp not terminal state
  const Uint nTotal = nRecurr + 2;

  input->prepare(traj, nTotal, samp - nRecurr, thrID);

  for(Uint k=0; k < 1+extraAlloc; k++)
    net->prepForBackProp(series[thrID + k*nThreads], nTotal);

  net->prepForFwdProp(series_tgt[thrID], nTotal);

  first_sample[thrID] = samp - nRecurr;
  thread_Wind[thrID] = wghtID;
  thread_seq[thrID] = traj;
}

Rvec Approximator::forward(const Uint samp, const Uint thrID,
  const int USE_WGT, const int USE_ACT, const int overwrite) const {
  if(USE_ACT>0) assert( (Uint) USE_ACT <= extraAlloc );
  // To handle Relay calling to answer agents' requests:
  if(thrID>=nThreads) return forward_agent(thrID-nThreads);

  const Uint netID = thrID + USE_ACT*nThreads;
  const vector<Activation*>& act = USE_ACT>=0? series[netID] :series_tgt[thrID];
  const vector<Activation*>& act_cur = series[thrID];
  const int ind = mapTime2Ind(samp, thrID);

  //if already computed just give answer
  if(act[ind]->written && not overwrite) return act[ind]->getOutput();

  // write previous outputs if needed (note: will spawn nested function calls)
  // previous output use the same weights only if not target weights
  if(ind>0 && not act_cur[ind-1]->written)
    forward(samp-1, thrID, std::max(USE_WGT, 0), 0);

  const Rvec inp = getInput(samp, thrID, USE_WGT);
  //cout <<"Input : "<< print(inp) << endl; fflush(0);
  return getOutput(inp, ind, act[ind], thrID, USE_WGT);
}

Rvec Approximator::getInput(const Uint samp, const Uint thrID, const int USEW) const {
  Rvec inp = input->forward(samp, thrID, USEW);
  if(relay not_eq nullptr) {
    const Rvec addedinp = relay->get(samp, thrID, USEW);
    assert(addedinp.size());
    inp.insert(inp.end(), addedinp.begin(), addedinp.end());
    //if(!thrID) cout << "relay "<<print(addedinp) << endl;
  }
  assert(inp.size() == net->getnInputs());
  return inp;
}

Rvec Approximator::getOutput(const Rvec inp, const int ind,
  Activation*const act, const Uint thrID, const int USEW) const {
  //hardcoded to use time series predicted with cur weights for recurrencies:
  const vector<Activation*>& act_cur = series[thrID];
  const Activation*const recur = ind? act_cur[ind-1] : nullptr;
  assert(USEW < (int) net->sampled_weights.size() );
  const Parameters* const W = opt->getWeights(USEW);
  assert( W not_eq nullptr );
  const Rvec ret = net->predict(inp, recur, act, W);
  //if(!thrID) cout<<"net fwd with inp:"<<print(inp)<<" out:"<<print(ret)<<endl;
  act->written = true;
  return ret;
}

void Approximator::applyImpSampling(Rvec& grad, const Sequence*const traj,
  const Uint samp) const
{
  const float anneal = std::min( (Real)1, opt->nStep * opt->epsAnneal);
  assert( anneal >= 0 );
  const float beta = 0.5 + 0.5 * anneal, P0 = traj->priorityImpW[samp];
  // if samples never seen by optimizer the samples have high priority
  // this matches one of last lines of Sampling::prepare()
  const float P = P0<=0 ? data->getMaxPriorityImpW() : P0;
  const float PERW = std::pow(data->getMinPriorityImpW() / P, beta);
  for(Uint i=0; i<grad.size(); i++) grad[i] *= PERW;
}

void Approximator::backward(Rvec grad, const Uint samp, const Uint thrID,
  const int USE_ACT) const
{
  if(USE_ACT>0) assert( (Uint) USE_ACT <= extraAlloc );
  const Uint netID = thrID + USE_ACT*nThreads;

  if( data->requireImpWeights )
    applyImpSampling(grad, thread_seq[thrID], samp);

  gradStats->track_vector(grad, thrID);
  const int ind = mapTime2Ind(samp, thrID);
  //ind+1 because we use c-style for loops in other places:
  error_placements[thrID] = std::max(ind+1, error_placements[thrID]);

  if(ESpopSize > 1) debugL("Skipping backward because we use ES.");

  const auto& act = USE_ACT>=0? series[netID] :series_tgt[thrID];
  assert(act[ind]->written);
  act[ind]->addOutputDelta(grad);
}

void Approximator::gradient(const Uint thrID, const int wID) const {
  if(error_placements[thrID]<=0) die("");

  nAddedGradients++;

  if(ESpopSize > 1)
  {
    debugL("Skipping gradient because we use ES (derivative-free) optimizers.");
  }
  else
  {
    const int last_error = error_placements[thrID];
    for(Uint j = 0; j<=extraAlloc; j++) {
      const Uint netID  = thrID +   j*nThreads;
      const Uint gradID = thrID + wID*nThreads;
      const vector<Activation*>& act = series[netID];
      for (int i=0; i<last_error; i++) assert(act[i]->written == true);

      net->backProp(act, last_error, net->Vgrad[gradID]);
      //for(int i=0;i<last_error&&!thrID;i++)cout<<i<<" inpG:"<<print(act[i]->getInputGradient(0))<<endl;
      if(input->net == nullptr || blockInpGrad) continue;

      for(int i=0; i<last_error; i++) {
        Rvec inputG = act[i]->getInputGradient(0);
        inputG.resize(input->nOutputs());
        input->backward(inputG, first_sample[thrID] +i, thrID);
      }
    }
  }
  error_placements[thrID] = -1; //to stop additional backprops
}

void Approximator::prepareUpdate()
{
  for(Uint i=0; i<nThreads; i++) if(error_placements[i]>0) die("");

  if(nAddedGradients==0) die("No-gradient update. Revise hyperparameters.");

  if(input->net not_eq nullptr and not blockInpGrad) {
    for(int i=0; i<ESpopSize; i++) input->losses[i] += losses[i];
  }

  opt->prepare_update(losses);
  losses = Rvec(ESpopSize, 0);
  reducedGradients = 1;
  nAddedGradients = 0;
}

void Approximator::applyUpdate() {
  if(reducedGradients == 0) return;

  opt->apply_update();
  net->updateTransposed();
  reducedGradients = 0;
}

void Approximator::getHeaders(ostringstream& buff) const {
  buff << std::left << std::setfill(' ') <<"| " << std::setw(6) << name;
  if(opt->tgtUpdateAlpha > 0) buff << "| dTgt ";
  opt->getHeaders(buff);
}

void Approximator::getMetrics(ostringstream& buff) const {
  real2SS(buff, net->weights->compute_weight_norm(), 7, 1);
  if(opt->tgtUpdateAlpha > 0)
    real2SS(buff, net->weights->compute_weight_dist(net->tgt_weights), 6, 1);
  opt->getMetrics(buff);
}

Rvec Approximator::relay_backprop(const Rvec err,
  const Uint samp, const Uint thrID, const bool bUseTargetWeights) const {
  if(relay == nullptr || relayInp < 0) die("improperly set up the relay");
  if(ESpopSize > 1) {
    debugL("Skipping relay_backprop because we use ES optimizers.");
    return Rvec(relay->nOutputs(), 0);
  }
  const vector<Activation*>& act = series_tgt[thrID];
  const int ind = mapTime2Ind(samp, thrID), nInp = input->nOutputs();
  assert(act[ind]->written == true && relay not_eq nullptr);
  const Parameters*const W = bUseTargetWeights? net->tgt_weights : net->weights;
  const Rvec ret = net->inpBackProp(err, act[ind], W, relayInp);
  for(Uint j=0; j<ret.size(); j++)
    assert(!std::isnan(ret[j]) && !std::isinf(ret[j]));
  //if(!thrID)
  //{
  //  const auto pret = Rvec(&ret[nInp], &ret[nInp+relay->nOutputs()]);
  //  const auto inp = act[ind]->getInput();
  //  const auto pinp = Rvec(&inp[nInp], &inp[nInp+relay->nOutputs()]);
  //  cout <<"G:"<<print(pret)<< " Inp:"<<print(pinp)<<endl;
  //}
  if(relayInp>0) return ret;
  else return Rvec(&ret[nInp], &ret[nInp+relay->nOutputs()]);
}

void Approximator::prepare_agent(Sequence*const traj, const Agent&agent,
  const Uint wghtID) const {
  //This is called by a std::thread and uses separate workspace from omp threads
  //We use a fake thread id to avoid code duplication in encapsulator class
  const Uint fakeThrID = nThreads + agent.ID, stepid = traj->ndata();
  agent_Wind[agent.ID] = wghtID;
  agent_seq[agent.ID] = traj;
  // learner->select always only gets one new state, so we assume that it needs
  // to run one (or more) forward net at time t, so here also compute recurrency
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT,stepid) : 0;
  const vector<Activation*>& act = agent_series[agent.ID];
  net->prepForFwdProp(agent_series[agent.ID], nRecurr+1);
  input->prepare(traj, nRecurr+1, stepid-nRecurr, fakeThrID);
  // if using relays, ask for previous actions, to be used for recurrencies
  // why? because the past is the past.
  if(relay not_eq nullptr) relay->prepare(traj, fakeThrID, ACT);
  assert(act.size() >= nRecurr+1);
  const Parameters* const W = opt->getWeights(wghtID);
  //Advance recurr net with 0 initialized activations for nRecurr steps
  for(Uint i=0, t=stepid-nRecurr; i<nRecurr; i++, t++)
    net->predict(getInput(t,fakeThrID,wghtID), i? act[i-1]:nullptr, act[i], W);
}

Rvec Approximator::forward_agent(const Uint agentID) const {
  // assume we already computed recurrencies
  const vector<Activation*>& act = agent_series[agentID];
  const int fakeThrID = nThreads + agentID, wghtID = agent_Wind[agentID];
  const Uint stepid = agent_seq[agentID]->ndata();
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, stepid) : 0;
  if(act[nRecurr]->written) return act[nRecurr]->getOutput();
  const Parameters* const W = opt->getWeights(wghtID);
  const Rvec inp = getInput(stepid, fakeThrID, wghtID);
  return net->predict(inp, nRecurr? act[nRecurr-1] : nullptr, act[nRecurr], W);
}

void Approximator::save(const string base, const bool bBackup) {
  if(opt == nullptr) die("Attempted to save uninitialized net!");
  opt->save(base + name, bBackup);
}
void Approximator::restart(const string base) {
  if(opt == nullptr) die("Attempted to restart uninitialized net!");
  opt->restart(base+name);
}
