//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "Encapsulator.h"
#include "Network.h"
#include "Optimizer.h"

Encapsulator::Encapsulator(const string N,const Settings&S,MemoryBuffer*const M)
: name(N), settings(S), data(M) { }

void Encapsulator::initializeNetwork(Network* _net, Optimizer* _opt)
{
  net = _net;
  opt = _opt;
  assert(opt not_eq nullptr && net not_eq nullptr);

  series.resize(nThreads);
  #pragma omp parallel for schedule(static, 1) num_threads(nThreads)
  for (Uint i=0; i<nThreads; i++) // numa aware allocation
   #pragma omp critical
    series[i].reserve(MAX_SEQ_LEN);

  if(not net->layers[0]->bInput) die("should not be possible");
  const Uint nInps = data->sI.dimUsed*(1+nAppended);
  if(net->layers[1]->spanCompInpGrads not_eq nInps)
    die("should not be possible");
  net->layers[1]->spanCompInpGrads  = 0;
  net->layers[1]->startCompInpGrads = nInps;
}

void Encapsulator::prepare(Sequence*const traj, const Uint len,
  const Uint samp, const Uint thrID) {
  thread_seq[thrID] = traj;
  if(net==nullptr) return;
  // before clearing out gradient, check if a backprop was ready
  // this should be performed when I place all gradients in the outputs
  // BUT depending on whether algorithm processes sequences/RNN
  // user might want to do backprop after placing many errors in outputs
  // instead of creating multiple functions to serve this purpose
  // order to perform backprop in implicit in re-allocation of work memory
  // and in order to advance the weights:
  if(error_placements[thrID] > 0) die("");
  first_sample[thrID] = samp;
  net->prepForBackProp(series[thrID], len);
  net->prepForFwdProp(series_tgt[thrID], len);
}

Rvec Encapsulator::state2Inp(const int t, const Uint thrID) const
{
  const Sequence*const traj = thread_seq[thrID];
  assert(t<(int)traj->tuples.size());
  const Uint nSvar = traj->tuples[t]->s.size();
  assert(nSvar == data->sI.dimUsed);
  if (nAppended>0)
  {
    vector <Real> inp((nAppended+1)*nSvar, 0);
    for(int k=t, j=0; j<=(int)nAppended; k--, j++)
    {
      const int kk = k<0 ? 0 : k; // copy multiple times s_0 at start of seq
      for(Uint i = 0; i < nSvar; i++)
      // j is fast index (different t of same feature are close, think CNN)
        inp[j + i*(nAppended+1)] = traj->tuples[kk]->s[i];
    }
    return data->standardizeAppended(inp);
  }
  else
  {
    debugS("encapsulate state %s", print(traj->tuples[t]->s).c_str() );
    return data->standardize(traj->tuples[t]->s);
  }
}

Rvec Encapsulator::forward(const int samp, const Uint thrID, const int wghtID) const
{
  if(net==nullptr) return state2Inp(samp, thrID);
  if(error_placements[thrID] > 0) die("");

  const vector<Activation*>&act = wghtID>=0? series[thrID]:series_tgt[thrID];
  const int ind = mapTime2Ind(samp, thrID);
  //if already computed just give answer
  if(act[ind]->written) return act[ind]->getOutput();
  const Parameters* const W = opt->getWeights(wghtID);
  act[ind]->written = true;
  return net->predict(state2Inp(samp, thrID), act[ind], W);
}

void Encapsulator::backward(const Rvec&error, const Uint samp, const Uint thrID) const
{
  if(net == nullptr) return;
  if(ESpopSize>1) die("should be impossible");
  const int ind = mapTime2Ind(samp, thrID);
  const vector<Activation*>& act = series[thrID];
  assert(act[ind]->written == true);
  //ind+1 because we use c-style for loops in other places:
  error_placements[thrID] = std::max(ind+1, error_placements[thrID]);
  act[ind]->addOutputDelta(error);
}

void Encapsulator::prepareUpdate()
{
  if(net == nullptr) return;

  for(Uint i=0; i<nThreads; i++) if(error_placements[i] > 0) die("");

  if(nAddedGradients==0) die("No-gradient update. Revise hyperparameters.");

  opt->prepare_update(losses);
  losses = Rvec(ESpopSize, 0);
  nReducedGradients = 1;
  nAddedGradients = 0;
}

void Encapsulator::applyUpdate()
{
  if(net == nullptr) return;
  if(nReducedGradients == 0) return;

  opt->apply_update();
  net->updateTransposed();
  nReducedGradients = 0;
}

void Encapsulator::gradient(const Uint thrID) const
{
  if(net == nullptr) return;

  nAddedGradients++;

  if(ESpopSize>1)
  {
    debugL("Skipping backprop because we use ES optimizers.");
  }
  else
  {
    if(error_placements[thrID]<=0) { warn("no input grad"); return;}

    const vector<Activation*>& act = series[thrID];
    const int last_error = error_placements[thrID];
    for (int i=0; i<last_error; i++) assert(act[i]->written == true);
    //if(!thrID) for(int i=0; i<last_error; i++)
    //  cout<<i<<" inpB:"<<print(act[i]->getOutputDelta())<<endl;
    net->backProp(act, last_error, net->Vgrad[thrID]);
  }
  error_placements[thrID] = -1; //to stop additional backprops
}

void Encapsulator::save(const string base, const bool bBackup)
{
  if(opt not_eq nullptr) opt->save(base+name, bBackup);
}
void Encapsulator::restart(const string base)
{
  if(opt not_eq nullptr) opt->restart(base+name);
}

void Encapsulator::getHeaders(ostringstream& buff) const
{
  if(net == nullptr) return;
  buff << std::left << std::setfill(' ') <<"| " << std::setw(5) << name;
  if(opt->tgtUpdateAlpha > 0) buff << "| dTgt ";
}

void Encapsulator::getMetrics(ostringstream& buff) const
{
  if(net == nullptr) return;
  real2SS(buff, net->weights->compute_weight_norm(), 6, 1);
  if(opt->tgtUpdateAlpha > 0)
    real2SS(buff, net->weights->compute_weight_dist(net->tgt_weights), 6, 1);
}
