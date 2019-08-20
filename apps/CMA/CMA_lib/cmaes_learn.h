#include "cmaes.h"
#include <random>

#include <string.h>
#include <chrono>
#include <algorithm>
#include <iostream>

#ifndef CMAES_LEARN_H
#define CMAES_LEARN_H



class Action
{
  public:
    const int dim;
    const int func_dim;
    std::vector<double> data = std::vector<double>(dim, 0);
    const double default_0;
    const double default_1;
    const double default_2;
    const double default_3;
    int lambda =  defaultLambda(func_dim);
    double lambda_frac = 1;
    Action(int _act_dim, int _func_dim, cmaes_t* const evo) :
    dim(_act_dim), func_dim(_func_dim), default_0(evo->sp.ccov1), default_1(evo->sp.ccovmu), default_2(evo->sp.ccumcov), default_3(evo->sp.cs)
    { }

    void update(  cmaes_t* const evo, double **arFunvals );

    static double zeroOneMap (double x) {
      return (1 + x/(1+std::fabs(x)) )/2;
    }
    static double zeroOneInv (double y) {
      return (2*y-1)/(0.5 - std::fabs(y - 0.5))/2;
    }

    static double zeroInfMap (double x) {
      return (x + std::sqrt(1+x*x));
    }
    static double zeroInfInv (double y) {
      return (y*y - 0.25)/y;
    }

    static int defaultLambda(const int _func_dim)
    {
      return std::floor( 4+std::floor( 3*std::log(_func_dim) ) );
    }
};


class State
{
  const int func_dim;

  public:

    static constexpr int dim = 6;
    std::vector<double> data = std::vector<double>(dim, 0);

    State(int _func_dim) : func_dim(_func_dim) { }

    void initial_state() {
      data = {1,1,0,0,(double)func_dim,0};
    }

    void final_state() {
      data = {0,0,0,0,(double)func_dim,0};
    }

    void update_state(cmaes_t*const evo, double&oldFmedian, double*oldXmean);
};


class write_cmaes_perf{

  public:
    void write( const int thrid );
    //void write( cmaes_t* const evo, const int thrid, int func_id, int step, const double final_dist, double ffinal );
    void write( cmaes_t* const evo,const int thrid, const int func_dim, int func_id, int step, const double final_dist, double ffinal );
};


void dump_curgen( double* const* pop, double *arFunvals, int step, int lambda, int func_dim );

void print_best_ever( cmaes_t* const evo, int func_dim );

void update_damps( cmaes_t* const evo );

int is_feasible(double* const pop, double* const lower_bound, double* const upper_bound, int dim);

void update_state(cmaes_t* const evo, double* const state, double* oldFmedian, double* oldXmean);

void copy_state( std::vector<double>& state, std::vector<double> from_state );

void actions_to_cma( double* const actions, int act_dim,  cmaes_t* const evo,
            int *lambda, double *lambda_fac, const int lambda_0, double **arFunvals );

bool check_for_nan_inf(cmaes_t* const evo, double* const* pop );

bool resample( cmaes_t* const evo, double* const* pop, double* const lower_bound, double* const upper_bound );


bool evaluate_and_update( cmaes_t* const evo, double* const*  pop, double *arFunvals, int* const info  );


void random_action( cmaes_t* const evo, std::mt19937 gen );

#endif
