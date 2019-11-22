//
//  main.cpp
//  cart-pole
//
//  Created by Dmitry Alexeev on 04/06/15.
//  Copyright (c) 2015 Dmitry Alexeev. All rights reserved.
//

#include "smarties.h"
#include "../cart_pole_cpp/cart-pole.h"

#include <iostream>
#include <cstdio>

inline int app_main(
  smarties::Communicator*const comm, // communicator with smarties
  MPI_Comm mpicom,         // mpi_comm that mpi-based apps can use
  int argc, char**argv    // arguments read from app's runtime settings file
)
{
  int myRank, simSize;
  MPI_Comm_rank(mpicom, & myRank);
  MPI_Comm_size(mpicom, & simSize);
  const int otherRank = myRank == 0? 1 : 0;
  assert(simSize == 2 && myRank < 2); // app designed to be run by 2 ranks

  comm->setStateActionDims(6, 1);

  //OPTIONAL: action bounds
  bool bounded = true;
  std::vector<double> upper_action_bound{10}, lower_action_bound{-10};
  comm->setActionScales(upper_action_bound, lower_action_bound, bounded);
  //OPTIONAL: hide angle, but not cosangle and sinangle.
  std::vector<bool> b_observable = {true, true, true, false, true, true};
  comm->setStateObservable(b_observable);

  CartPole env;

  MPI_Barrier(mpicom);
  while(true) //train loop
  {
    //reset environment:
    env.reset(comm->getPRNG());
    comm->sendInitState(env.getState()); //send initial state

    while (true) //simulation loop
    {
      //advance the simulation:
      const std::vector<double> action = comm->recvAction();

      int terminated[2] = {0, 0};
      terminated[myRank] = env.advance(action);
      MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT,
                    terminated, 1, MPI_INT, mpicom);
      const bool myEnvTerminated = terminated[myRank];
      const bool otherTerminated = terminated[otherRank];

      const std::vector<double> state = env.getState();
      const double reward = env.getReward();

      // Environment simulation is distributed across two processes.
      // Still, if one processes says the simulation has terminated
      // it should terminate in all processes! (and then can start anew)
      if(myEnvTerminated || otherTerminated) {
        if(myEnvTerminated) comm->sendTermState(state, reward);
        else comm->sendLastState(state, reward);
        break;
      }
      else comm->sendState(state, reward);
    }
  }
}

int main(int argc, char**argv)
{
  smarties::Engine e(argc, argv);
  if( e.parse() ) return 1;
  // this app is designed to require 2 processes per each env simulation:
  e.setNworkersPerEnvironment(2);
  e.run( app_main );
  return 0;
}