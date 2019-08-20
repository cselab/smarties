
#include "smarties.h"
#include "pp.h"

#include <cstdio>

inline void app_main(smarties::Communicator*const comm, int argc, char**argv)
{
  const unsigned maxStep = 500;
  comm->set_num_agents(2); // predator prey
  comm->agents_define_different_MDP(); // pred & prey learn different policies
  //Sim box has size EXTENT. Fraction of box that agent can traverse in 1 step:
  const double velScale = 0.02 * EXTENT;
  comm->set_state_action_dims(4, 2, 0); // 4 state, 2 control variables
  comm->set_state_action_dims(4, 2, 1); // 4 state, 2 control variables

  // predator additional arg is how much slower than prey (eg 50%)
  Predator pred(4, velScale, 0.5);
  // prey arg is observation noise (eg ping of predator is in 1 stdev of noise)
  Prey     prey(4, velScale, 1.0);

  std::mt19937& gen = comm->getPRNG(); // different seed on each process
  Window plot;

  unsigned sim = 0;
  while(true) //train loop
  {
    //reset environment:
    pred.reset(gen); //comm contains rng with different seed on each rank
    prey.reset(gen); //comm contains rng with different seed on each rank

    //send initial state
    comm->sendInitState(pred.getState(prey),      0);
    comm->sendInitState(prey.getState(pred, gen), 1);

    unsigned step = 0;
    while (true) //simulation loop
    {
      pred.advance(comm->recvAction(0));
      prey.advance(comm->recvAction(1));

      plot.update(step, sim, pred.p[0], pred.p[1], prey.p[0], prey.p[1]);

      if(step++ < maxStep)
      {
        comm->sendState(pred.getState(prey),      pred.getReward(prey), 0);
        comm->sendState(prey.getState(pred, gen), prey.getReward(pred), 1);
      }
      else
      {
        comm->sendLastState(pred.getState(prey),      pred.getReward(prey), 0);
        comm->sendLastState(prey.getState(pred, gen), prey.getReward(pred), 1);
        sim++;
        break;
      }
    }
  }
  return 0;
}

int main(int argc, char**argv)
{
  smarties::Engine e(argc, argv);
  if( e.parse() ) return 1;
  e.run( app_main );
  return 0;
}