//
//  main.cpp
//  cart-pole
//
//  Created by Dmitry Alexeev on 04/06/15.
//  Copyright (c) 2015 Dmitry Alexeev. All rights reserved.
//

#include "smarties.h"
#include "../cart_pole_cpp/cart-pole.h"

#include <cstdio>
#define DIFFERENT_MDPS

inline int app_main(
  smarties::Communicator*const comm, int argc, char**argv
)
{
  comm->setNumAgents(2);
  comm->setStateActionDims(6, 1);
  const std::vector<double> upper_action_bound{10}, lower_action_bound{-10};
  comm->setActionScales(upper_action_bound, lower_action_bound, true);
  // state vars :                        x    vx  angvel  ang   cos    sine
  const std::vector<bool> bObservable1{true, true, true, false, true, true};
  comm->setStateObservable(bObservable1);

  #ifdef DIFFERENT_MDPS
    // All information about the MDP before calling
    // agents_define_different_MDP() is copied over to all agents.
    comm->agentsDefineDifferentMDP();
    // one agent is partially observed: linear and angular vels are hidden
    const std::vector<bool> bObservable2{true, false, false, false, true, true};
    comm->setStateObservable(bObservable2, 1);
    comm->setIsPartiallyObservable(1);
  #endif
  // Moreover, agent 0 will have inverted controls relative to agent 1

  // Here for simplicity we have two environments
  // But real application is one env with competing/collaborating agents
  CartPole env1, env2;

  while(true) //train loop
  {
    //reset environment:
    env1.reset(comm->getPRNG()); //comm contains rng with different seed on each rank
    env2.reset(comm->getPRNG()); //comm contains rng with different seed on each rank

    comm->sendInitState(env1.getState(), 0); //send initial state
    comm->sendInitState(env2.getState(), 1); //send initial state

    while (true) //simulation loop
    {
      std::vector<double> action1 = comm->recvAction(0);
      #ifdef DIFFERENT_MDPS
        action1[0] = - action1[0]; // make the two optimal policy different
      #endif
      std::vector<double> action2 = comm->recvAction(1);

      //advance the simulation:
      const bool terminated1 = env1.advance(action1);
      const bool terminated2 = env2.advance(action2);

      const std::vector<double> state1 = env1.getState();
      const std::vector<double> state2 = env2.getState();
      const double reward1 = env1.getReward();
      const double reward2 = env2.getReward();

      if(terminated1 || terminated2)  //tell smarties this is a terminal state
      {
        if(terminated1) comm->sendTermState(state1, reward1, 0);
        else comm->sendLastState(state1, reward1, 0);
        if(terminated2) comm->sendTermState(state2, reward2, 1);
        else comm->sendLastState(state2, reward2, 1);
        break;
      }
      else
      {
        comm->sendState(state1, reward1, 0);
        comm->sendState(state2, reward2, 1);
      }
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
