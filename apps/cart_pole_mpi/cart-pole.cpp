//
//  main.cpp
//  cart-pole
//
//  Created by Dmitry Alexeev on 04/06/15.
//  Copyright (c) 2015 Dmitry Alexeev. All rights reserved.
//

#include "smarties.h"
#include "../cart_pole_cpp/cart-pole.h"
//#include "../../source/Utils/Profiler.h"

#include <iostream>
#include <cstdio>

inline void app_main(smarties::Communicator*const comm, // communicator with smarties
                    MPI_Comm mpicom,         // mpi_comm that mpi-based apps can use
                    int argc, char**argv    // arguments read from app's runtime settings file
)
{
  comm->set_state_action_dims(6, 1);
  //smarties::Profiler prof;
  //int wrank;
  //MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
  //OPTIONAL: action bounds
  bool bounded = true;
  std::vector<double> upper_action_bound{10}, lower_action_bound{-10};
  comm->set_action_scales(upper_action_bound, lower_action_bound, bounded);
  
  /*
   // ALTERNATIVE for discrete actions:
   std::vector<int> n_options = vector<int>{2};
   comm.set_action_options(n_options);
   // will receive either 0 or 1, app chooses resulting outcome
   */
  
  //OPTIONAL: hide state variables.
  // e.g. show cosine/sine but not angle
  std::vector<bool> b_observable = {true, true, true, false, true, true};
  //std::vector<bool> b_observable = {true, false, false, false, true, true};
  comm->set_state_observable(b_observable);
  
  //OPTIONAL: set space bounds
  std::vector<double> upper_state_bound{ 1,  1,  1,  1,  1,  1};
  std::vector<double> lower_state_bound{-1, -1, -1, -1, -1, -1};
  comm->set_state_scales(upper_state_bound, lower_state_bound);
  // Here for simplicity we have two environments
  // But real application is to env with two competing/collaborating agents
  CartPole env;
  int sim_id = 0;
  while(true) //train loop
  {
    sim_id ++;
    //prof.start("env");
    env.reset(comm->getPRNG()); //comm contains rng with different seed on each rank
    const auto S0 = env.getState();
    //prof.stop_start("smarties");
    comm->sendInitState(S0); //send initial state
    if(comm->terminateTraining()) return; // exit program

    while (true) //simulation loop
    {
      std::vector<double> action = comm->recvAction();

      //prof.stop_start("env");
      const bool terminated = env.advance(action);
      const std::vector<double> state = env.getState();
      const double reward = env.getReward();

      //prof.stop_start("smarties");
      if(terminated) { //tell smarties that this is a terminal state
        comm->sendTermState(state, reward);
        //prof.stop();
        //break;
      } else comm->sendState(state, reward);

      if(comm->terminateTraining()) return; // exit program
      if(terminated) break; // go back up to reset
    }
    //if(sim_id % 100 == 0)
    //  printf("Rank %d\n%s\n", wrank, prof.printStatAndReset().c_str() );
  }
  return;
}

int main(int argc, char**argv)
{
  smarties::Engine e(argc, argv);
  if( e.parse() ) return 1;
  e.run( app_main );
  return 0;
}
