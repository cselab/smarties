#include "smarties.h"
#include "glider.h"

inline void app_main(smarties::Communicator*const comm)
{
  std::cout << "Glider with density ratio " << RHORATIO <<
   " and aspect ratio = " << ASPECTRATIO << ". Instantaneous reward is " <<
   (INSTREW == 0 ? "mixed" : (INSTREW == 1 ? "time" : "energy"));
  #ifdef SPEED_PENAL
  std::cout << " with penalization on terminal velocity";
  #endif
  std::cout << ".\n";

  comm->set_state_action_dims(10, 1);
  std::mt19937& gen = comm->getPRNG();

  bool bounded = true;
  std::vector<double> upper_action_bound{1}, lower_action_bound{-1};
  comm->set_action_scales(upper_action_bound, lower_action_bound, bounded);
  std::vector<bool> b_observable = {1, 1, 1, 1, 1, 1, 1, 0, 0, 0};
  //vector<bool> b_observable = {0, 0, 0, 1, 1, 1, 1, 0, 0, 0};
  comm->set_state_observable(b_observable);
  comm->finalize_problem_description();

  Glider env;

  #ifdef USE_SMARTIES
  while (true) //train loop
  #endif
  {
    //reset environment:
    env.reset(gen); //comm contains rng with different seed on each rank
    #ifdef USE_SMARTIES //send initial state:
      comm->sendInitState(env.getState(gen));
    #else // read initial conditions from command line:
      env.set({stod(argv[1]), stod(argv[2]), stod(argv[3]),
               stod(argv[4]), stod(argv[5]), stod(argv[6])});
    #endif

    while (true) //simulation loop
    {
      #ifdef USE_SMARTIES
        std::vector<double> action = comm->recvAction();
      #else
        std::vector<double> action = {0};
      #endif

      //advance the simulation:
      bool terminated = env.advance(action);
      std::vector<double> state = env.getState(gen);
      double reward = env.getReward();

      #ifndef USE_SMARTIES
        std::cout<<env._s.u<<" "<<env._s.v<<" "<<env._s.w
                 <<" "<<env._s.x<<" "<<env._s.y<<" "<<env._s.a<<std::endl;
      #endif

      if(terminated)  //tell smarties that this is a terminal state
      {
        #ifdef USE_SMARTIES
          comm->sendTermState(state, env.getTerminalReward());
        #endif
        {
          env.updateOldDistanceAndEnergy();
          FILE * pFile = fopen ("terminals.raw", "ab");
          const int writesize = 9*sizeof(float);
          float* buf = (float*) malloc(writesize);
          buf[0] = env.time; buf[1] = env.oldEnergySpent;
          buf[2] = env._s.x; buf[3] = env._s.y; buf[4] = env._s.a;
          buf[5] = env._s.u; buf[6] = env._s.v; buf[7] = env._s.w;
          buf[8] = env.getTerminalReward();
          fwrite (buf, sizeof(float), writesize/sizeof(float), pFile);
          fflush(pFile); fclose(pFile);  free(buf);
        }
        break;
      }
      #ifdef USE_SMARTIES
      else comm->sendState(state, reward);
      #endif
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