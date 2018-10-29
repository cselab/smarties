//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "../ReplayMemory/MemoryBuffer.h"
#include "Network.h"

class Builder;
class Optimizer;
class Network;

struct Encapsulator
{
  const std::string name;
  const Settings& settings;
  const Uint nThreads = settings.nThreads+settings.nAgents;
  const Uint nAppended = settings.appendedObs;
  const int ESpopSize = settings.ESpopSize;

  THRvec<std::vector<Activation*>> series =
                                THRvec<std::vector<Activation*>>(nThreads);
  THRvec<std::vector<Activation*>> series_tgt =
                                THRvec<std::vector<Activation*>>(nThreads);
  THRvec<int> first_sample = THRvec<int>(nThreads, -1);
  THRvec<int> error_placements = THRvec<int>(nThreads, -1);
  THRvec<Sequence*> thread_seq = THRvec<Sequence*>(nThreads, nullptr);

  // For CMAES based optimization. Keeps track of total loss associate with
  // Each weight vector sample:
  mutable Rvec losses = Rvec(ESpopSize, 0);

  mutable std::atomic<Uint> nAddedGradients{0};
  Uint nReducedGradients = 0;
  MemoryBuffer* const data;
  Optimizer* opt = nullptr;
  Network* net = nullptr;

  inline Uint nOutputs() const {
    if(net==nullptr) return data->sI.dimUsed*(1+nAppended);
    else return net->getnOutputs();
  }

  Encapsulator(const string N,const Settings&S,MemoryBuffer*const M);

  void initializeNetwork(Network* _net, Optimizer* _opt);

  void prepare(Sequence*const traj, const Uint len, const Uint samp, const Uint thrID);

  inline int mapTime2Ind(const Uint samp, const Uint thrID) const {
    assert(first_sample[thrID]<=(int)samp);
    //ind is mapping from time stamp along trajectoy and along alloc memory
    const int ind = (int)samp - first_sample[thrID];
    return ind;
  }

  Rvec state2Inp(const int t, const Uint thrID) const;

  Rvec forward(const int samp, const Uint thrID, const int wghtID) const;

  void backward(const Rvec&error, const Uint samp, const Uint thrID) const;

  void prepareUpdate();

  void applyUpdate();

  void gradient(const Uint thrID) const;

  void save(const std::string base, const bool bBackup);
  void restart(const std::string base = std::string());

  void getHeaders(std::ostringstream& buff) const;
  void getMetrics(std::ostringstream& buff) const;
};
