//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "mpi.h"
#include "../Communicators/Communicator.h"

#include <iostream>
#include <cmath>
#include <cassert>
#include <sstream>

#include <random>
std::string printableTuple(std::vector<double> s,double r,std::vector<double> a);
int app_main(Communicator*const rlcom, MPI_Comm mpicom, int argc, char**argv, const Uint numSteps);

std::string printableTuple(std::vector<double> s,double r,std::vector<double> a)
{
  std::ostringstream o;
  o << "[";
  for (int i=0; i<static_cast<int>(s.size())-1; i++) o << s[i] << " ";
  o << s[s.size()-1] << "] " << r << " [";
  for (int i=0; i<static_cast<int>(a.size())-1; i++) o << a[i] << " ";
  o << a[a.size()-1] << "]";
  return o.str();
}

int app_main(Communicator*const rlcom, MPI_Comm mpicom, int argc, char**argv, const unsigned numSteps)
{
  //std::ostringstream o;
  //o << argc << ":";
  //for (int i=0; i<argc; ++i) o << " [" << argv[i] << "]";
  //printf("%s\n",o.str().c_str()); fflush(0);
  const int nS = rlcom->getStateDim();
  const int nA = rlcom->getActionDim();
  int rank, size, wrank;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-1.73205, 1.73205); //mean0 var1
  MPI_Comm_rank(mpicom, &rank);
  MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
  MPI_Comm_size(mpicom, &size);
  std::cout << rank << " " << size << std::endl;
  std::vector<double> state(nS);
  std::vector<double> action(nA);
  double* vec = (double*) malloc((nS+1)*sizeof(double));
  unsigned steps = 0;
  while(steps<numSteps)
  {
    int status = 1;

    for(int k=0; k<10; k++)
    {
      if(k==9) status = 2;

      if(rank == size-1) { //if last rank, generate state
        for(int i=0; i<nS+1; i++) vec[i] = dist(gen);
      } else {
        MPI_Recv(vec, nS+1, MPI_DOUBLE, rank+1, 1337, mpicom, MPI_STATUS_IGNORE);
      }
      if(rank) //if not rank 0, pass state along
        MPI_Send(vec, nS+1, MPI_DOUBLE, rank-1, 1337, mpicom);

      for(int i=0; i<nS; i++) state[i] = vec[i];
      double reward = vec[nS];

      rlcom->sendState(0, status, state, reward);
      if(status != 2) {
        action = rlcom->recvAction();
        steps++;
      }

      //printf("Rank %d (%d) comm %s\n",
      //  rank, wrank, printableTuple(state, reward, action).c_str());
      //fflush(0);
      status = 0;
    }
  }
  free(vec);
  return 0;
}
