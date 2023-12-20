//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_MPIUtilities_h
#define smarties_MPIUtilities_h

#include <mpi.h>
#include <omp.h>
#include <stdexcept>

namespace smarties
{

inline MPI_Comm MPICommDup(const MPI_Comm C) {
  if(C == MPI_COMM_NULL) return MPI_COMM_NULL;
  MPI_Comm ret;
  MPI_Comm_dup(C, &ret);
  return ret;
}
inline unsigned MPICommSize(const MPI_Comm C) {
  if(C == MPI_COMM_NULL) return 0;
  int size;
  MPI_Comm_size(C, &size);
  return (unsigned) size;
}
inline unsigned MPICommRank(const MPI_Comm C) {
  if(C == MPI_COMM_NULL) return 0;
  int rank;
  MPI_Comm_rank(C, &rank);
  return (unsigned) rank;
}
inline unsigned MPIworldRank() { return MPICommRank(MPI_COMM_WORLD); }

#ifdef REQUIRE_MPI_MULTIPLE
  #define MPI(NAME, ...)                                   \
  do {                                                     \
    /*int MPIERR =*/ MPI_ ## NAME ( __VA_ARGS__ );             \
    /*if(MPIERR not_eq MPI_SUCCESS) {         */               \
    /*  _warn("%s %d", #NAME, MPIERR);        */               \
    /*  throw std::runtime_error("MPI ERROR");*/               \
    /*}                                       */               \
  } while(0)
#else
  #define MPI(NAME, ...)                                   \
  do {                                                     \
    int MPIERR = 0;                                        \
    if(distrib.bAsyncMPI) {                                \
      MPIERR = MPI_ ## NAME ( __VA_ARGS__ );               \
    } else {                                               \
      std::lock_guard<std::mutex> lock(distrib.mpiMutex);  \
      MPIERR = MPI_ ## NAME ( __VA_ARGS__ );               \
    }                                                      \
    if(MPIERR not_eq MPI_SUCCESS) {                        \
      _warn("%s %d", #NAME, MPIERR);                       \
      Warnings::print_stacktrace();                        \
      throw std::runtime_error("MPI ERROR");               \
    }                                                      \
  } while(0)
#endif


} // end namespace smarties
#endif // smarties_MPIUtilities_h
