//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_DelayedReductor_h
#define smarties_DelayedReductor_h

#include "../Settings/ExecutionInfo.h"

#include <mutex>

namespace smarties
{

template<typename T>
struct DelayedReductor
{
  const MPI_Comm mpicomm;
  const Uint arysize, mpisize;
  const ExecutionInfo & distrib;
  MPI_Request buffRequest = MPI_REQUEST_NULL;
  std::vector<T> return_ret = std::vector<T>(arysize, 0);
  std::vector<T> reduce_ret = std::vector<T>(arysize, 0);
  std::vector<T> partialret = std::vector<T>(arysize, 0);

  static int getSize(const MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
  }

  DelayedReductor(const ExecutionInfo &, const std::vector<T> init);
  ~DelayedReductor();

  std::vector<T> get(const bool accurate = false);

  template<Uint I>
  T get(const bool accurate = false)
  {
    const std::vector<T> ret = get(accurate);
    return ret[I];
  }

  void beginRDX();
  void update(const std::vector<T> ret);
};

}
#endif
