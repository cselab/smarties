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

#define NCARTS 2

inline int app_main(
  smarties::Communicator*const comm, // communicator with smarties
  MPI_Comm mpicom,         // mpi_comm that mpi-based apps can use
  int argc, char**argv    // arguments read from app's runtime settings file
)
{
  int myRank, simSize;
  MPI_Comm_rank(mpicom, & myRank);
  MPI_Comm_size(mpicom, & simSize);
  assert(simSize==NCARTS && myRank<NCARTS); // app runs with NCARTS ranks

  // This options says that the agent themselves are distributed.
  // I.e. the same agent runs on multiple ranks:
  comm->env_has_distributed_agents();
  // Because we are holding on to using cart-poles... let's just say that our
  // agent is NCARTS cart-poles with joint controls. 4 state and 1 control
  // variables per process, distributed over NCARTS processes.
  comm->set_state_action_dims(4 * NCARTS, 1 * NCARTS);

  //OPTIONAL: action bounds
  const bool bounded = true;
  const std::vector<double> upper_action_bound(NCARTS,  10);
  const std::vector<double> lower_action_bound(NCARTS, -10);
  comm->set_action_scales(upper_action_bound, lower_action_bound, bounded);

  CartPole env;

  MPI_Barrier(mpicom);
  while(true) //train loop
  {
    {
      //reset environment:
      env.reset(comm->getPRNG());
      const std::vector<double> myState = env.getState(4);
      std::vector<double> combinedState = std::vector<double>(4 * NCARTS);

      MPI_Allgather(      myState.data(), 4, MPI_DOUBLE,
                    combinedState.data(), 4, MPI_DOUBLE, mpicom);
      // Actually, only rank 0 will send the state to smarties.
      // We might as well have used MPI_Gather with root 0.
      comm->sendInitState(combinedState);
    }

    while (true) //simulation loop
    {
      // Each rank will get the same vector here:
      const std::vector<double> combinedAction = comm->recvAction();
      assert(combinedAction.size() == NCARTS);
      const std::vector<double> myAction = { combinedAction[myRank] };

      const int myTerminated = env.advance(myAction);
      const std::vector<double> myState = env.getState(4);
      const double myReward = env.getReward();

      std::vector<double> combinedState = std::vector<double>(4 * NCARTS);
      double sumReward = 0;
      int nTerminated = 0;

      MPI_Allreduce(&myTerminated, &nTerminated, 1, MPI_INT, MPI_SUM, mpicom);
      MPI_Allreduce(&myReward, &sumReward, 1, MPI_DOUBLE, MPI_SUM, mpicom);
      MPI_Allgather(      myState.data(), 4, MPI_DOUBLE,
                    combinedState.data(), 4, MPI_DOUBLE, mpicom);

      // Environment simulation is distributed across two processes.
      // Still, if one processes says the simulation has terminated
      // it should terminate in all processes! (and then can start anew)
      if(nTerminated > 0) {
        comm->sendTermState(combinedState, sumReward);
        break;
      }
      else comm->sendState(combinedState, sumReward);
    }
  }
}

int main(int argc, char**argv)
{
  smarties::Engine e(argc, argv);
  if( e.parse() ) return 1;
  // this app is designed to require NCARTS processes per each env simulation:
  e.setNworkersPerEnvironment(NCARTS);
  e.run( app_main );
  return 0;
}