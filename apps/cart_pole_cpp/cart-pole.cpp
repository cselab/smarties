//
//  main.cpp
//  cart-pole
//
//  Created by Dmitry Alexeev on 04/06/15.
//  Copyright (c) 2015 Dmitry Alexeev. All rights reserved.
//

#include "smarties.h"
#include "cart-pole.h"

#include <iostream>
#include <cstdio>

inline void app_main(smarties::Communicator*const comm, int argc, char**argv)
{
  const int control_vars = 1; // force along x
  const int state_vars = 6; // x, y, angvel, angle, cosine, sine
  comm->set_state_action_dims(state_vars, control_vars);

  //OPTIONAL: action bounds
  bool bounded = true;
  std::vector<double> upper_action_bound{10}, lower_action_bound{-10};
  comm->set_action_scales(upper_action_bound, lower_action_bound, bounded);

  /*
    // ALTERNATIVE for discrete actions:
    vector<int> n_options = vector<int>{2};
    comm->set_action_options(n_options);
    // will receive either 0 or 1, app chooses resulting outcome
  */

  //OPTIONAL: hide state variables.
  // e.g. show cosine/sine but not angle
  std::vector<bool> b_observable = {true, true, true, false, true, true};
  comm->set_state_observable(b_observable);

  //OPTIONAL: set space bounds
  std::vector<double> upper_state_bound{ 1,  1,  1,  1,  1,  1};
  std::vector<double> lower_state_bound{-1, -1, -1, -1, -1, -1};
  comm->set_state_scales(upper_state_bound, lower_state_bound);

  CartPole env;

  while(true) //train loop
  {
    env.reset(comm->getPRNG()); // prng with different seed on each process
    comm->sendInitState(env.getState()); //send initial state

    while (true) //simulation loop
    {
      std::vector<double> action = comm->recvAction();
      if(comm->terminateTraining()) return; // exit program

      bool poleFallen = env.advance(action); //advance the simulation:

      std::vector<double> state = env.getState();
      double reward = env.getReward();

      if(poleFallen) { //tell smarties that this is a terminal state
        comm->sendTermState(state, reward);
        break;
      } else comm->sendState(state, reward);
    }
  }
}

int main(int argc, char**argv)
{
  smarties::Engine e(argc, argv);
  if( e.parse() ) return 1;
  e.run( app_main );
  return 0;
}
