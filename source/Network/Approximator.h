//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Encapsulator.h"

struct Aggregator;

enum NET { CUR, TGT }; /* use CUR or TGT weights */
struct Approximator
{
  const Settings& settings;
  const std::string name;
  const bool bRecurrent = settings.bRecurrent;
  const Uint nAgents = settings.nAgents, nThreads = settings.nThreads;
  const Uint mpisize = settings.learner_size, nMaxBPTT = settings.nnBPTTseq;
  const int ESpopSize = settings.ESpopSize;
  Encapsulator* const input;
  MemoryBuffer* const data;
  const Aggregator* const relay;
  Optimizer* opt = nullptr;
  Network* net = nullptr;
  StatsTracker* gradStats = nullptr;

  THRvec<int> error_placements = THRvec<int>(nThreads, -1);
  THRvec<int> first_sample     = THRvec<int>(nThreads, -1);
  THRvec<int> agent_Wind       = THRvec<int>(nAgents, -1);
  THRvec<int> thread_Wind      = THRvec<int>(nThreads, -1);
  THRvec<Sequence*> agent_seq  = THRvec<Sequence*>(nAgents, nullptr);
  THRvec<Sequence*> thread_seq = THRvec<Sequence*>(nThreads, nullptr);

  mutable std::atomic<Uint> nAddedGradients{0};
  Uint reducedGradients=0;
  int relayInp = -1;
  Uint extraAlloc = 0;

  // whether to backprop gradients in the input network.
  // work by DeepMind (eg in D4PG) indicates it's best to not propagate
  // policy net gradients towards input conv layers
  bool blockInpGrad = false;

  //thread safe memory for prediction with current weights:
  THRvec<std::vector<Activation*>> series =
                                     THRvec<std::vector<Activation*>>(nThreads);

  //thread safe agent specific activations
  THRvec<std::vector<Activation*>> agent_series =
                                     THRvec<std::vector<Activation*>>(nAgents);

  //thread safe  memory for prediction with target weights. Rules are that
  // index along the two alloc vectors is the same for the same sample, and
  // that tgt net (if available) takes recurrent activation from current net:
  THRvec<std::vector<Activation*>> series_tgt =
                                     THRvec<std::vector<Activation*>>(nThreads);

  // For CMAES based optimization. Keeps track of total loss associate with
  // Each weight vector sample:
  mutable Rvec losses = Rvec(ESpopSize, 0);

  Approximator(const string _name, Settings& S, Encapsulator*const en,
    MemoryBuffer* const data_ptr, const Aggregator* const r = nullptr);
  ~Approximator();

  Builder buildFromSettings(Settings& _s, const vector<Uint> n_outputs);
  Builder buildFromSettings(Settings& _s, const Uint n_outputs);

  void initializeNetwork(Builder& build);
  void allocMorePerThread(const Uint nAlloc);

  // agent always uses mean weight vector, also when using evol strategies
  void prepare_agent(Sequence* const traj, const Agent& agent,
    const Uint wghtID=0) const;
  Rvec forward_agent(const Uint aID) const;
  inline Rvec forward_agent(const Agent&agent) const {
    return forward_agent(agent.ID);
  }

  void prepare_seq(Sequence*const traj, const Uint thrID,
    const Uint wghtID) const;

  void prepare_one(Sequence*const traj, const Uint samp,
      const Uint thrID, const Uint wghtID) const;
  void prepare(Sequence*const traj, const Uint samp, const Uint N,
      const Uint thrID, const Uint wghtID) const;

  Rvec forward(const Uint samp, const Uint thrID,
    const int USE_WGT, const int USE_ACT, const int overwrite=0) const;
  inline Rvec forward(const Uint samp, const Uint thrID, int USE_ACT=0) const {
    assert(USE_ACT>=0);
    return forward(samp, thrID, thread_Wind[thrID], USE_ACT);
  }
  template<NET USE_A = CUR>
  inline Rvec forward_cur(const Uint samp, const Uint thrID) const {
    const int indA = USE_A==CUR? 0 : -1;
    return forward(samp, thrID, thread_Wind[thrID], indA);
  }
  template<NET USE_A = TGT>
  inline Rvec forward_tgt(const Uint samp, const Uint thrID) const {
    const int indA = USE_A==CUR? 0 : -1;
    return forward(samp, thrID, -1, indA);
  }

  inline Rvec get(const Uint samp, const Uint thrID, int USE_ACT=0) const {
    const Uint netID = thrID + USE_ACT*nThreads;
    const vector<Activation*>&act = USE_ACT>=0? series[netID]:series_tgt[thrID];
    return act[mapTime2Ind(samp, thrID)]->getOutput();
  }

  // relay backprop requires gradients: no wID, no sorting based opt algos
  Rvec relay_backprop(const Rvec grad, const Uint samp, const Uint thrID,
    const bool bUseTargetWeights = false) const;

  void backward(Rvec grad, const Uint samp, const Uint thrID, const int USE_ACT=0) const;

  void gradient(const Uint thrID, const int wID = 0) const;

  void prepareUpdate();

  void applyUpdate();

  inline Uint nOutputs() const {
   return net->getnOutputs();
  }

  void getMetrics(ostringstream& buff) const;
  void getHeaders(ostringstream& buff) const;

  void save(const string base, const bool bBackup);
  void restart(const string base = string());

  inline void updateGradStats(const string base, const Uint iter) const {
    gradStats->reduce_stats(base+name, iter);
  }

 private:
  Rvec getOutput(const Rvec inp, const int ind,
    Activation*const act, const Uint thrID, const int USEW) const;

  Rvec getInput(const Uint samp, const Uint thrID, const int USEW) const;

  inline int mapTime2Ind(const Uint samp, const Uint thrID) const {
    assert(first_sample[thrID]<=(int)samp);
    //ind is mapping from time stamp along trajectoy and along alloc memory
    const int ind = (int)samp - first_sample[thrID];
    return ind;
  }

  void applyImpSampling(Rvec& G, const Sequence*const S, const Uint t) const;
};
