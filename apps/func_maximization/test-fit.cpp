//
//  main.cpp
//  cart-pole
//
//  Created by Dmitry Alexeev on 04/06/15.
//  Copyright (c) 2015 Dmitry Alexeev. All rights reserved.
//

#include <cmath>
#include <random>
#include <cstdio>
#include <vector>
#include <functional>
#include "smarties.h"

// two options for running this file:
// 1) find the coefficients of the 2D ROSENBROCK func (1, 100)
// 2) find the minimum of the 2D ROSENBROCK (1, 1)
//#define FIND_PARAM

inline void app_main(smarties::Communicator*const comm, int argc, char**argv)
{
  const int control_vars = 2; // parameters to find
  const int state_vars = 0; // nothing : function maximization
  comm->setStateActionDims(state_vars, control_vars);

  #ifdef FIND_PARAM
   const std::vector<double> upper_act_scale{1, 100}, lower_act_scale{-1, -100};
  #else
   const std::vector<double> upper_act_scale{1, 1}, lower_act_scale{-1, -1};
  #endif
  comm->setActionScales(upper_act_scale, lower_act_scale, false);
  std::uniform_real_distribution<double> dist(-2, 2);
  auto & G = comm->getPRNG();

  const double A = 1;
  const double B = 100;
  const auto F = [](const double x, const double y,
                    const double a, const double b) {
     return std::pow(x-a, 2) + b * std::pow(y - x*x, 2);
  };

  while (true)
  {
    const std::vector<double> params = comm->getOptimizationParameters();
    assert(params.size() == 2);
    #ifdef FIND_PARAM
      const double Y = dist(G), X = dist(G);
      const double Z = F(X, Y, A, B);
      const double z = F(X, Y, params[0], params[1]);
      const double R = - std::pow(Z-z, 2);
    #else
      const double Z = F(params[0], params[1], A, B);
      const double R = - Z;
    #endif

    comm->setOptimizationEvaluation(R);
  }
}

int main(int argc, char**argv)
{
  smarties::Engine e(argc, argv);
  if( e.parse() ) return 1;
  e.run( app_main );
  return 0;
}

