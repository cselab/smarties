/* --------------------------------------------------------- */
/* --------------- A Very Short Example -------------------- */
/* --------------------------------------------------------- */

#define _XOPEN_SOURCE 500
#define _BSD_SOURCE
#define __RLON 1
#define __RANDACT 0
#define __NGENSKIP 1
#include "cmaes_interface.h"
#include "Communicator.h"
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <sstream>
#define VERBOSE 0
/*#define _RESTART_*/
#define _IODUMP_ 0
#define JOBMAXTIME  0

#include "cmaes_learn.h"

#include "fitfun.h"


/* the optimization loop */
int main(int argn, char **args)
{
  static constexpr int actinDim = 4;
  static constexpr int stateDim = State::dim;

  if (argn<1) {
    printf("Did not receive the socket. Aborting.\n");
    abort();
  }

  const int sock = std::stoi(args[1]);
  std::mt19937 generator(sock);

  //std::normal_distribution<double>      func_dim_distribution(0, 3);
  std::uniform_int_distribution<int>     func_dim_distribution(1,10);
  std::uniform_real_distribution<double>   start_x_distribution(.3,.7);
  std::uniform_real_distribution<double>   start_std_distribution(.2,.5);
  std::uniform_int_distribution<int>     func_ID_distribution(0, _COUNT-1);
  //subset of functions:
  //std::uniform_int_distribution<int>     func_ID_distribution(7,8);
  std::uniform_int_distribution<int>     cma_seed_distribution(0,std::numeric_limits<int>::max());

  #if __RLON
    //communicator class, it needs a socket number sock, given by RL as first argument of execution
    Communicator comm(sock, stateDim, actinDim);
  #endif

  const int thrid = 0; //omp_get_thread_num();

  write_cmaes_perf wcp;

  wcp.write(thrid);

  while (true)
  {
    int step = 0; // cmaes stepping
    int info[4]; //legacy: gen, chain, step, task
    cmaes_t * const evo = new cmaes_t(); /* a CMA-ES type struct or "object" */
    double oldFmedian, *oldXmean = nullptr; //related to RL rewards
    double *lower_bound, *upper_bound, *init_x, *init_std; //IC for cmaes
    double *arFunvals, *const*pop;  //cma current function values and samples

    //const int func_dim = 1+ceil(std::fabs(func_dim_distribution(generator )));
    const int func_dim = func_dim_distribution( generator );
    const int runseed  = cma_seed_distribution( generator );

    info[0] = func_ID_distribution( generator );

    evo->sp.funcID = info[0];
    printf("Selected function %d with dimensionality %d\n", info[0], func_dim);

    init_x       = (double*)malloc(func_dim * sizeof(double));
    init_std     = (double*)malloc(func_dim * sizeof(double));
    lower_bound = (double*)malloc(func_dim * sizeof(double));
    upper_bound = (double*)malloc(func_dim * sizeof(double));

    get_upper_lower_bounds(lower_bound, upper_bound, func_dim, info);

    for (int i = 0; i < func_dim; i++) { //to be returned from function?
      init_x[i] = start_x_distribution( generator )
              *(upper_bound[i]-lower_bound[i]) + lower_bound[i];
      init_std[i] = start_std_distribution( generator )
              *(upper_bound[i]-lower_bound[i]);
    }

    arFunvals = cmaes_init(evo, func_dim, init_x, init_std, runseed,
      Action::defaultLambda(func_dim), "../cmaes_initials.par");
    printf("%s\n", cmaes_SayHello(evo));
    cmaes_ReadSignals(evo, "../cmaes_signals.par");  /* write header and initial values */

    //vectors of states and actions
    State states(func_dim);
    Action actions(actinDim, func_dim, evo);

    #if __RLON
    {
      states.initial_state();
      comm.sendInitState(states.data);
      actions.data = comm.recvAction();
      actions.update( evo, &arFunvals );
    }
    #elif __RANDACT
      random_action(evo, generator )
    #endif

    bool bConverged = false;
    while(true) // Iterate until stop criterion holds
    {
      // actions are constant in this loop
      for(int dG = 0; dG < __NGENSKIP*func_dim; dG++)
      {
        // generate lambda new search points, check for nans
        pop = cmaes_SamplePopulation(evo);

        bConverged = check_for_nan_inf( evo, pop );

        // re-sample if not feasible, check if stuck
        if( !bConverged)
          bConverged = resample( evo, pop, lower_bound, upper_bound );

        // evaluate current pop and update distribution
        if(!bConverged)
          bConverged = evaluate_and_update(evo, pop, arFunvals, info  );

        if(!bConverged){
          if (step == 0) { //need an initial state
            oldFmedian = cmaes_Get(evo, "fmedian");
            oldXmean = cmaes_GetNew(evo, "xmean");
          }

          if(cmaes_TestForTermination(evo)) {
            bConverged = true;
          }
        }

        step += actions.lambda;

        fflush(stdout); // useful in MinGW

        if(bConverged) break;

        #if VERBOSE
          print_best_ever( evo, step );
        #endif

        #if _IODUMP_
          dump_curgen( pop, arFunvals, step, actions.lambda, func_dim );
        #endif
      } // end of constant action loop

      if (bConverged) break; //go to send terminal state

      #if __RLON
      {
        states.update_state( evo, oldFmedian, oldXmean );
        comm.sendState(states.data, -.01*actions.lambda_frac);
        actions.data = comm.recvAction();
        actions.update( evo, &arFunvals );
      }
      #elif __RANDACT
        random_action(evo, generator );
      #endif

    } // end of single function optimization

    if (evo->isStuck == 1)
    {
      fprintf(stderr, "Stopping becoz stuck\n");

      states.final_state();

      for (int i = 0; i < actions.lambda; ++i) {
        std::ostringstream o;
        o << "[";
        for (int j=0; j<func_dim; j++) {
          o << pop[i][j];
          if (i < func_dim-1) o << " ";
        }
        o << "]";
        printf("Evaluated function in %s = %e\n",o.str().c_str(),arFunvals[i]);
      }
      #if __RLON
        comm.sendTermState(states.data, -1.);
      #endif
    }
    else
    {
      states.update_state( evo, oldFmedian, oldXmean);

      double* xfinal = cmaes_GetNew(evo, "xmean");
      double ffinal;

      fitfun(xfinal, func_dim, &ffinal, info);

      const double final_dist = eval_distance_from_optimum(xfinal, func_dim, info);
      const double r_end     = std::max(-1., 1-1e2*final_dist);

      #if __RLON
        comm.sendTermState(states.data, r_end);
      #endif

      wcp.write( evo, thrid,func_dim, info[0], step, final_dist, ffinal );

      free(xfinal);
    }

    printf("Stop: %s\n",  cmaes_TestForTermination(evo)); /* print termination reason */

    //cmaes_WriteToFile(&evo, "all", "allcmaes.dat");         /* write final results */
    cmaes_exit(evo); /* release memory */
    delete evo;
    if(oldXmean not_eq nullptr) free(oldXmean);
    free(lower_bound); free(upper_bound);
    free(init_x); free(init_std);

  } // end of learning loop

  return 0;
}
