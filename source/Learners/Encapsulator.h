//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "MemoryBuffer.h"
#include "../Network/Network.h"
#include "../Network/Optimizer.h"
#include <list>

class Builder;

enum OUTPUT { STATE, STATEACT }; /* use CUR or TGT weights */
//template <OUTPUT OUTP = STATE>
struct Encapsulator
{
  const string name;
  const Uint nThreads, nAppended;
  Settings& settings;
  mutable vector<vector<Activation*>> series;
  mutable vector<int> first_sample;
  mutable vector<int> error_placements;
  mutable Uint nAddedGradients=0, nReducedGradients = 0;
  MemoryBuffer* const data;
  Optimizer* opt = nullptr;
  Network* net = nullptr;

  inline Uint nOutputs() const {
    if(net==nullptr) return data->sI.dimUsed*(1+nAppended);
    else return net->getnOutputs();
  }

  Encapsulator(const string _name, Settings& sett, MemoryBuffer*const data_ptr)
  : name(_name), nThreads(sett.nThreads+sett.nAgents),
    nAppended(sett.appendedObs), settings(sett), first_sample(nThreads,-1),
    error_placements(nThreads,-1), data(data_ptr) {}

  void initializeNetwork(Network* _net, Optimizer* _opt)
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

  inline void prepare(const Uint len, const Uint samp, const Uint thrID) const
  {
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
  }

  inline int mapTime2Ind(const Uint samp, const Uint thrID) const
  {
    assert(first_sample[thrID]<=(int)samp);
    //ind is mapping from time stamp along trajectoy and along alloc memory
    const int ind = (int)samp - first_sample[thrID];
    return ind;
  }

  inline Rvec state2Inp(const int t, const Sequence*const traj,
      const Uint thrID) const
  {
    assert(t<(int)traj->tuples.size());
    const Uint nSvar = traj->tuples[t]->s.size();
    assert(nSvar == data->sI.dimUsed);
    if (nAppended>0) {
      vector <Real> inp((nAppended+1)*nSvar, 0);
      for(int k=t, j=0; j<=(int)nAppended; k--, j++)
      {
        const int kk = k<0 ? 0 : k; // copy multiple times s_0 at start of seq
        for(Uint i = 0; i < nSvar; i++)
        // j is fast index (different t of same feature are close, think CNN)
          inp[j + i*(nAppended+1)] = traj->tuples[kk]->s[i];
      }
      #ifdef NOISY_INPUT
        //if(traj->ID>=0) return data->standardizeAppendedNoisy(traj->tuples[t]->s, thrID);
      #endif
      return data->standardizeAppended(inp);
    } else
    #ifdef NOISY_INPUT
      if(traj->ID>=0) return data->standardizeNoisy(traj, t, thrID);
    #endif
    debugS("encapsulate state %s", print(traj->tuples[t]->s).c_str() );
    return data->standardize(traj->tuples[t]->s);
  }

  inline Rvec forward(const Sequence*const seq, const int samp,
    const Uint thrID) const
  {
    if(net==nullptr) return state2Inp(samp, seq, thrID); //data->Tmp[agentId]);
    if(error_placements[thrID] > 0) die("");

    const vector<Activation*>& act = series[thrID];
    const int ind = mapTime2Ind(samp, thrID);
    //if already computed just give answer
    if(act[ind]->written == true) return act[ind]->getOutput();
    const Rvec inp = state2Inp(samp, seq, thrID);
    assert(inp.size() == net->getnInputs());
    const Rvec ret = net->predict(inp, act[ind]);
    act[ind]->written = true;
    return ret;
  }

  inline void backward(const Rvec&error, const Uint samp,
    const Uint thrID) const
  {
    if(net == nullptr) return;
    const int ind = mapTime2Ind(samp, thrID);
    const vector<Activation*>& act = series[thrID];
    assert(act[ind]->written == true);
    //ind+1 because we use c-style for loops in other places:
    error_placements[thrID] = std::max(ind+1, error_placements[thrID]);
    act[ind]->addOutputDelta(error);
  }

  void prepareUpdate(const Uint batchSize)
  {
    if(net == nullptr) return;

    for(Uint i=0; i<nThreads; i++) if(error_placements[i] > 0) die("");

    if(nAddedGradients==0) die("No-gradient update. Revise hyperparameters.");
    if(nAddedGradients>batchSize) die("weird");

    opt->prepare_update(batchSize, net->Vgrad);
    nReducedGradients = nAddedGradients;
    nAddedGradients = 0;
  }

  void applyUpdate()
  {
    if(net == nullptr) return;
    if(nReducedGradients == 0) return;

    opt->apply_update();
    net->updateTransposed();
    nReducedGradients = 0;
  }

  inline void gradient(const Uint thrID) const
  {
    if(net == nullptr) return;
    if(error_placements[thrID]<=0) { warn("no input grad"); return;}

    #pragma omp atomic
    nAddedGradients++;

    vector<Activation*>& act = series[thrID];
    const int last_error = error_placements[thrID];
    for (int i=0; i<last_error; i++) assert(act[i]->written == true);
    //if(!thrID) for(int i=0; i<last_error; i++)
    //  cout<<i<<" inpB:"<<print(act[i]->getOutputDelta())<<endl;
    net->backProp(act, last_error, net->Vgrad[thrID]);
    error_placements[thrID] = -1; //to stop additional backprops
  }

  void save(const string base = string())
  {
    if(opt not_eq nullptr) opt->save(base+name);
  }
  void restart(const string base = string())
  {
    if(opt not_eq nullptr) opt->restart(base+name);
  }

  void getHeaders(ostringstream& buff) const
  {
    if(net == nullptr) return;
    buff << std::left << std::setfill(' ') <<"| " << std::setw(5) << name;
  }

  void getMetrics(ostringstream& buff) const
  {
    if(net == nullptr) return;
    real2SS(buff, net->weights->compute_weight_norm(), 6, 1);
  }
};


#if 0
void Learner_utils::dumpPolicy()
{
  //a fail in any of these amounts to a big and fat TODO
  if(nAppended) die("TODO missing features\n");
  const Uint nDumpPoints = env->getNdumpPoints();
  const Uint n_outs = 4;
  printf("n_outs:%u, nInputs:%u, nDumpPoints:%u\n",n_outs,nInputs, nDumpPoints);
  FILE * pFile = fopen ("dump.raw", "wb");
  Rvec output(nOutputs);
  vector<float> dump(nInputs+n_outs);
  Activation* act = net->allocateActivation();
  for (Uint i=0; i<nDumpPoints; i++)
  {
    Rvec state = env->getDumpState(i);
    assert(state.size()==nInputs);
    net->predict(data->standardize(state), output, act);
    Uint k=0;
    for (Uint j=0; j<nInputs; j++) dump[k++] = state[j];
    for (Uint j=0; j<n_outs; j++) dump[k++] = output[j];
    //state.insert(state.end(),output.begin(),output.end()); //unsafe
    fwrite(dump.data(),sizeof(float),dump.size(),pFile);
  }
  _dispose_object(act);
  fclose (pFile);
}

void Learner_utils::dumpNetworkInfo(const int agentId) const
{
  #ifdef _dumpNet_
  if (bTrain) return;
  #else
  return;
  #endif
  net->dump(agentId);
  Rvec output(nOutputs);
  const Uint ndata = data->Tmp[agentId]->tuples.size(); //last one already placed
  if (ndata == 0) return;

  vector<Activation*> series_base = net->allocateUnrolledActivations(ndata);
  for (Uint k=0; k<ndata; k++) {
    const Tuple * const _t = data->Tmp[agentId]->tuples[k];
    net->predict(data->standardize(_t->s), output, series_base, k);
  }

  string fname="gradInputs_"+to_string(agentId)+"_"+to_string(ndata)+".dat";
  ofstream out(fname.c_str());
  if (!out.good()) _die("Unable to open save into file %s\n", fname.c_str());
  //first row of file is net output:
  for(Uint j=0; j<nOutputs; j++) out << output[j] << " ";
  out << "\n";
  //sensitivity of value for this action in this state wrt all previous inputs
  Uint start0 = ndata > MAX_UNROLL_BFORE ? ndata-MAX_UNROLL_BFORE-1 : 0;
  for (Uint ii=start0; ii<ndata; ii++) {
    Uint start1 = ii > MAX_UNROLL_BFORE ? ii-MAX_UNROLL_BFORE-1 : 0;
    for (Uint i=0; i<nInputs; i++) {
      vector<Activation*> series =net->allocateUnrolledActivations(ndata);
      for (Uint k=start1; k<ndata; k++) {
        Rvec state = data->Tmp[agentId]->tuples[k]->s;
        if (k==ii) state[i] = 0;
        net->predict(data->standardize(state), output, series, k);
      }
      Rvec oDiff = net->getOutputs(series.back());
      Rvec oBase = net->getOutputs(series_base.back());
      //other rows of file are d(net output)/d(state_i(t):
      for(Uint j=0; j<nOutputs; j++) {
        const Real dOut = oDiff[j]-oBase[j];
        const Real dState = data->Tmp[agentId]->tuples[ii]->s[i];
        out << dOut/dState << " ";
      }
      out << "\n";
      net->deallocateUnrolledActivations(&series);
    }
  }
  out.close();
  net->deallocateUnrolledActivations(&series_base);
}
#endif
/*
    #if defined(ExpTrust) || !defined(GradClip) //then trust region is computed on batch
      if (thrID==0) net->backProp(series_cur, nRecurr, net->grad);
      else net->backProp(series_cur, nRecurr, net->Vgrad[thrID]);
    #else           //else trust region on this temp gradient
      net->backProp(series_cur, nRecurr, Ggrad[thrID]);
    #endif

    #ifdef GradClip
      net->prepForBackProp(series_1[thrID], nRecurr);
      Rvec trust = grad_kldiv(seq, samp, policies[0]);

        net->setOutputDeltas(trust, series_cur[nRecurr-1]);
        net->backProp(series_cur, nRecurr, Kgrad[thrID]);

      #ifndef ExpTrust
        if (thrID==0) circle_region(Ggrad[thrID], Kgrad[thrID], net->grad, DKL_target);
        else circle_region(Ggrad[thrID], Kgrad[thrID], net->Vgrad[thrID], DKL_target);
        //if (thrID==0) fullstats(Ggrad[thrID], Kgrad[thrID], net->grad, DKL_target);
        //else fullstats(Ggrad[thrID], Kgrad[thrID], net->Vgrad[thrID], DKL_target);
      #endif
    #endif

    //#ifdef FEAT_CONTROL
    //  const Uint task_out0 = ContinuousSignControl::addRequestedLayers(nA,
    //    env->sI.dimUsed, net_indices, net_outputs, out_weight_inits);
    // task = new ContinuousSignControl(task_out0, nA, env->sI.dimUsed, net,data);
    //#endif
    //#ifdef FEAT_CONTROL
    // const Rvec act=aInfo.getInvScaled(data->Set[seq]->tuples[samp]->a);
    // const Activation*const recur = nSValues>1 ? series_hat[1] : nullptr;
    // task->Train(series_cur.back(), recur, act, seq, samp, grad);
    //#endif

        #if defined(ExpTrust) || !defined(GradClip)  //then trust region is computed on batch
        #else           //else trust region on this temp gradient
          net->backProp(series_cur, ndata-1, Ggrad[thrID]);
        #endif

        #ifdef GradClip
          net->prepForBackProp(series_1[thrID], ndata-1);
          for (int k=static_cast<int>(ndata)-2; k>=0; k--) {
            const Tuple * const _t = data->Set[seq]->tuples[k];
            Policy_t pol = prepare_policy(net->getOutputs(series_cur[k]));
            pol.prepare(_t->a, _t->mu, bGeometric, nullptr);
            Rvec trust = grad_kldiv(seq, k, pol);
            net->setOutputDeltas(trust, series_cur[k]);
          }

          net->backProp(series_cur, ndata-1, Kgrad[thrID]);

          #ifndef ExpTrust
            if (thrID==0) circle_region(Ggrad[thrID], Kgrad[thrID], net->grad, DKL_target);
            else circle_region(Ggrad[thrID], Kgrad[thrID], net->Vgrad[thrID], DKL_target);
          #endif
        #endif

        //#ifdef FEAT_CONTROL
        //const Rvec act=aInfo.getInvScaled(data->Set[seq]->tuples[k]->a);
        //task->Train(series_cur[k], series_hat[k+1], act, seq, k, grad);
        //#endif
  #ifdef ExpTrust
  void stackAndUpdateNNWeights() override
  {
    if(!nAddedGradients) die("Error in stackAndUpdateNNWeights\n");
    assert(bTrain);
    opt->nepoch++;
    Uint nTotGrads = nAddedGradients;
    opt->stackGrads(Kgrad[0], Kgrad);
    opt->stackGrads(net->grad, net->Vgrad); //add up gradients across threads
    if (learn_size > 1) { //add up gradients across masters
      MPI_Allreduce(MPI_IN_PLACE, Kgrad[0]->_W, net->getnWeights(),
          MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
      MPI_Allreduce(MPI_IN_PLACE, Kgrad[0]->_B, net->getnBiases(),
          MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
      MPI_Allreduce(MPI_IN_PLACE, net->grad->_W, net->getnWeights(),
          MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
      MPI_Allreduce(MPI_IN_PLACE, net->grad->_B, net->getnBiases(),
          MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
      MPI_Allreduce(MPI_IN_PLACE,&nTotGrads,1,MPI_UNSIGNED,MPI_SUM,mastersComm);
    }

    circle_region(Kgrad[0], net->grad, DKL_target, nTotGrads);
    //update is deterministic: can be handled independently by each node
    //communication overhead is probably greater than a parallelised sum
    opt->update(net->grad, nTotGrads);
  }
  #endif

  inline Rvec grad_kldiv(const Uint seq, const Uint samp, const Policy_t& pol_cur) const
  {
    const Tuple * const _t = data->Set[seq]->tuples[samp];
    const Rvec gradDivKL = pol_cur.div_kl_opp_grad(_t->mu, 1);
    Rvec gradient(nOutputs,0);
    pol_cur.finalize_grad(gradDivKL, gradient);
    //clip_gradient(gradient, stdGrad[0], seq, samp);
    return gradient;
  }
  //for (auto & trash : Kgrad) _dispose_object(trash);
  Kgrad.resize(nThreads); Ggrad.resize(nThreads);
  #pragma omp parallel for
  for (Uint i=0; i<nThreads; i++) {
   Kgrad[i] = new Grads(net->getnWeights(), net->getnBiases());
   Ggrad[i] = new Grads(net->getnWeights(), net->getnBiases());
  }
*/
