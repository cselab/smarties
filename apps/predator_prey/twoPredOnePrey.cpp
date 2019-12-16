#include <cstdio>
#include "smarties.h"

// DEFINES BEFORE INCLUDE twoPredOnePrey.h:
#define EXTENT 1.0
#define dt 1.0
#define SAVEFREQ 2000
#define STEPFREQ 1
//#define PERIODIC
#define COOPERATIVE
//#define PLOT_TRAJ

#include "twoPredOnePrey.h"

inline void app_main(smarties::Communicator*const comm, int argc, char**argv)
{
  const unsigned maxStep = 500;
  comm->setNumAgents(3); // predator prey
  comm->agentsDefineDifferentMDP(); // pred & prey learn different policies
  //Sim box has size EXTENT. Fraction of box that agent can traverse in 1 step:
  const double maxSpeed = 0.02 * EXTENT/dt;
  comm->setStateActionDims(6, 2, 0); // 6 state, 2 control variables
  comm->setStateActionDims(6, 2, 1); // 6 state, 2 control variables
  comm->setStateActionDims(6, 2, 2); // 6 state, 2 control variables

  std::mt19937 &rngPointer =  comm->getPRNG();
 
  // predator last arg is how much slower than prey (eg 50%)
  Predator pred1(rngPointer, 6, maxSpeed, 0.5);
  Predator pred2(rngPointer, 6, maxSpeed, 0.5);
  // prey last arg is observation noise (eg ping of predator is in 1 stdev of noise)
  // Prey     prey(rngPointer, 6, maxSpeed, 1.0); // The noise was large, the prey didn't run away quickly if preds were far away
  Prey     prey(rngPointer, 6, maxSpeed, 0.0);

  #ifdef COOPERATIVE
    printf("Cooperative predators\n");
  #else
    printf("Competitive predators\n");
  #endif
  fflush(NULL);

  #ifdef PLOT_TRAJ
    Window plot;
  #endif

  unsigned sim = 0;
  while(true) //train loop
  {
    //reset environment:
    pred1.reset();
    pred2.reset();
    prey.reset();

    //send initial state
    comm->sendInitState(pred1.getState(prey,pred2), 0);
    comm->sendInitState(pred2.getState(prey,pred1), 1);
    comm->sendInitState(prey.getState(pred1,pred2), 2);

    unsigned step = 0;
    while (true) //simulation loop
    {
      pred1.advance(comm->recvAction(0));
      pred2.advance(comm->recvAction(1));
      prey.advance(comm->recvAction(2));

  	  std::vector<bool> gotCaught = prey.checkTermination(pred1,pred2);
  	  if(prey.is_over()) // Terminate simulation
      { 
  		  // Cooperative hunting - both predators get reward
  		  const double finalReward = 10*EXTENT;

        #ifdef COOPERATIVE
    		  comm->sendTermState(pred1.getState(prey,pred2), finalReward, 0);
    		  comm->sendTermState(pred2.getState(prey,pred1), finalReward, 1);
        #else
    		  // Competitive hunting - only one winner, other one gets jack (also, 
          // change the reward to be not d1*d2, but just d_i if use competitive)
          comm->sendTermState(pred1.getState(prey,pred2), finalReward*gotCaught[0], 0);
          comm->sendTermState(pred2.getState(prey,pred1), finalReward*gotCaught[1], 1);
        #endif
  		  comm->sendTermState(prey.getState(pred1,pred2),-finalReward, 2);

  		  printf("Sim #%d reporting that prey got its world rocked.\n", sim);
        fflush(NULL);
  		  sim++; break;
  	  }

      #ifdef PLOT_TRAJ
        plot.update(step, sim, pred1.p, pred2.p, prey.p);
      #endif

      if(step++ < maxStep)
      {
        comm->sendState(  pred1.getState(prey,pred2), pred1.getReward(prey,pred2), 0);
        comm->sendState(  pred2.getState(prey,pred1), pred2.getReward(prey,pred1), 1);
        comm->sendState(  prey.getState(pred1,pred2), prey.getReward(pred1,pred2), 2);
      }
      else
      {
        comm->sendLastState(pred1.getState(prey,pred2), pred1.getReward(prey,pred2), 0);
        comm->sendLastState(pred2.getState(prey,pred1), pred2.getReward(prey,pred2), 1);
        comm->sendLastState(prey.getState(pred1,pred2), prey.getReward(pred1,pred2), 2);
        sim++;
        break;
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
