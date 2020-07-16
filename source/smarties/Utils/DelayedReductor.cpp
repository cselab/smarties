//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "DelayedReductor.h"
#include "Warnings.h"
#include "SstreamUtilities.h"
#include "../Settings/Bund.h"

#include <cassert>

namespace smarties
{

template<typename T>
DelayedReductor<T>::DelayedReductor(const ExecutionInfo& D,
                                    const std::vector<T> I) :
mpicomm(MPICommDup(D.learners_train_comm)), arysize(I.size()),
mpisize(MPICommSize(D.learners_train_comm)), distrib(D), return_ret(I) {}


template<typename T>
DelayedReductor<T>::~DelayedReductor()
{
  MPI_Comm* commptr = const_cast<MPI_Comm *>(&mpicomm);
  MPI_Comm_free(commptr);
}

template<typename T>
std::vector<T> DelayedReductor<T>::get(const bool accurate)
{
  if(buffRequest not_eq MPI_REQUEST_NULL) {
    int completed = 0;
    if(accurate) {
      completed = 1;
      MPI(Wait, &buffRequest, MPI_STATUS_IGNORE);
    } else {
      MPI(Test, &buffRequest, &completed, MPI_STATUS_IGNORE);
    }
    if( completed ) {
      return_ret = reduce_ret;
      buffRequest = MPI_REQUEST_NULL;
    }
  }
  return return_ret;
}

template<typename T>
void DelayedReductor<T>::update(const std::vector<T> ret)
{
  assert(ret.size() == arysize);
  if (mpisize <= 1) {
    buffRequest = MPI_REQUEST_NULL;
    return_ret = ret;
    return;
  }

  if(buffRequest not_eq MPI_REQUEST_NULL) {
    MPI(Wait, &buffRequest, MPI_STATUS_IGNORE);
    buffRequest = MPI_REQUEST_NULL;
    return_ret = reduce_ret;
  }
  reduce_ret = ret;
  assert(mpicomm not_eq MPI_COMM_NULL);
  assert(buffRequest == MPI_REQUEST_NULL);
  beginRDX();
}

template<> void DelayedReductor<long double>::beginRDX()
{
  MPI(Iallreduce, MPI_IN_PLACE, reduce_ret.data(), arysize,
                 MPI_LONG_DOUBLE, MPI_SUM, mpicomm, &buffRequest);
}
template<> void DelayedReductor<long>::beginRDX()
{
  MPI(Iallreduce, MPI_IN_PLACE, reduce_ret.data(), arysize,
                 MPI_LONG, MPI_SUM, mpicomm, &buffRequest);
}

template struct DelayedReductor<long>;
template struct DelayedReductor<long double>;


}
