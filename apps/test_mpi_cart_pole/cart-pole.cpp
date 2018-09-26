//
//  main.cpp
//  cart-pole
//
//  Created by Dmitry Alexeev on 04/06/15.
//  Copyright (c) 2015 Dmitry Alexeev. All rights reserved.
//

#include <iostream>
#include <cmath>
#include <random>
#include <cstdio>
#include <vector>
#include <functional>

#include "mpi.h"
#include "Communicator.h"
#define SWINGUP 0
using namespace std;

// Julien Berland, Christophe Bogey, Christophe Bailly,
// Low-dissipation and low-dispersion fourth-order Runge-Kutta algorithm,
// Computers & Fluids, Volume 35, Issue 10, December 2006, Pages 1459-1463, ISSN 0045-7930,
// http://dx.doi.org/10.1016/j.compfluid.2005.04.003
template <typename Func, typename Vec>
Vec rk46_nl(double t0, double dt, Vec u0, Func&& Diff)
{
  const double a[] = {0.000000000000, -0.737101392796, -1.634740794341, -0.744739003780, -1.469897351522, -2.813971388035};
  const double b[] = {0.032918605146,  0.823256998200,  0.381530948900,  0.200092213184,  1.718581042715,  0.270000000000};
  const double c[] = {0.000000000000,  0.032918605146,  0.249351723343,  0.466911705055,  0.582030414044,  0.847252983783};

  const int s = 6;
  Vec w;
  Vec u(u0);
  double t;

  #pragma unroll
    for (int i=0; i<s; i++)
    {
      t = t0 + dt*c[i];
      w = w*a[i] + Diff(u, t)*dt;
      u = u + w*b[i];
    }
  return u;
}

struct Vec4
{
  double y1, y2, y3, y4;

  Vec4(double y1=0, double y2=0, double y3=0, double y4=0) : y1(y1), y2(y2), y3(y3), y4(y4) {};

  Vec4 operator*(double v) const
  {
    return Vec4(y1*v, y2*v, y3*v, y4*v);
  }

  Vec4 operator+(const Vec4& v) const
  {
    return Vec4(y1+v.y1, y2+v.y2, y3+v.y3, y4+v.y4);
  }
};

struct CartPole
{
  const double mp = 0.1;
  const double mc = 1;
  const double l = 0.5;
  const double g = 9.81;
  const double dt = 4e-4;
  const int nsteps = 50;
  int info=1, step=0;
  Vec4 u;
  double F=0, t=0;

	void reset(std::mt19937& gen) {
		#if SWINGUP
	    std::uniform_real_distribution<double> dist(-.1,.1);
		#else
	    std::uniform_real_distribution<double> dist(-0.05,0.05);
		#endif
		u = Vec4(dist(gen), dist(gen), dist(gen), dist(gen));
		F = t = step = 0;
		info = 1;
	}

  bool is_over() {
    #if SWINGUP
      return step>=500 || std::fabs(u.y1)>2.4;
    #else
      return step>=500 || std::fabs(u.y1)>2.4 || std::fabs(u.y3)>M_PI/15;
    #endif
  }

  int advance(vector<double> action) {
    F = action[0];
    step++;
    for (int i=0; i<nsteps; i++) {
      u = rk46_nl(t, dt, u, bind(&CartPole::Diff, this, placeholders::_1, placeholders::_2) );
      t += dt;
      if( is_over() ) return 1;
    }
    return 0;
  }

	vector<double> getState() {
    vector<double> state(6);
		state[0] = u.y1;
		state[1] = u.y2;
		state[2] = u.y4;
		state[3] = u.y3;
		state[4] = std::cos(u.y3);
		state[5] = std::sin(u.y3);
		return state;
	}

  double getReward()
  {
    #if SWINGUP
  		double angle = std::fmod(u.y3, 2*M_PI);
  		angle = angle<0 ? angle+2*M_PI : angle;
  		return fabs(angle-M_PI)<M_PI/6 ? 1 : 0;
    #else
      return 1 - ( std::fabs(u.y3)>M_PI/15 || std::fabs(u.y1)>2.4 );
		#endif
  }

  Vec4 Diff(Vec4 u, double t)
  {
    Vec4 res;

    const double cosy = std::cos(u.y3);
    const double siny = std::sin(u.y3);
    const double w = u.y4;

    #if SWINGUP
			const double fac1 = 1./(mc + mp * siny*siny);
			const double fac2 = fac1/l;
			res.y2 = fac1*(F + mp*siny*(l*w*w + g*cosy));
			res.y4 = fac2*(-F*cosy -mp*l*w*w*cosy*siny -(mc+mp)*g*siny);
    #else
      const double totMass = mp+mc;
      const double fac2 = l*(4./3. - mp*cosy*cosy/totMass);
      const double F1 = F + mp * l * w * w * siny;
      res.y4 = (g*siny - F1*cosy/totMass)/fac2;
      res.y2 = (F1 - mp*l*res.y4*cosy)/totMass;
    #endif
    res.y1 = u.y2;
    res.y3 = u.y4;
    return res;
  }
};

int app_main(
  Communicator*const comm, // communicator with smarties
  MPI_Comm mpicom,         // mpi_comm that mpi-based apps can use
  int argc, char**argv,    // arguments read from app's runtime settings file
  const unsigned numSteps      // number of time steps to run before exit
) {
  comm->update_state_action_dims(6, 1);

  //OPTIONAL: action bounds
  bool bounded = true;
  vector<double> upper_action_bound{10}, lower_action_bound{-10};
  comm->set_action_scales(upper_action_bound, lower_action_bound, bounded);

  /*
    // ALTERNATIVE for discrete actions:
    vector<int> n_options = vector<int>{2};
    comm.set_action_options(n_options);
    // will receive either 0 or 1, app chooses resulting outcome
  */

  //OPTIONAL: hide state variables.
  // e.g. show cosine/sine but not angle
  vector<bool> b_observable = {true, true, true, false, true, true};
  //vector<bool> b_observable = {true, false, false, false, true, true};
  comm->set_state_observable(b_observable);

  //OPTIONAL: set space bounds
  vector<double> upper_state_bound{ 1,  1,  1,  1,  1,  1};
  vector<double> lower_state_bound{-1, -1, -1, -1, -1, -1};
  comm->set_state_scales(upper_state_bound, lower_state_bound);
  // Here for simplicity we have two environments
  // But real application is to env with two competing/collaborating agents
  CartPole env;

  while(true) //train loop
  {
    //reset environment:
    env.reset(comm->getPRNG()); //comm contains rng with different seed on each rank


    comm->sendInitState(env.getState()); //send initial state

    while (true) //simulation loop
    {
      vector<double> action = comm->recvAction();

      //advance the simulation:
      bool terminated = env.advance(action);

      vector<double> state = env.getState();
      double reward = env.getReward();

      if(terminated)  //tell smarties that this is a terminal state
      {
        comm->sendTermState(state, reward);
        break;
      }
      else comm->sendState(state, reward);
    }
  }
  return 0;
}
