#include <iostream>
#include <cmath>
#include <random>
#include <cstdio>
#include <vector>
#include <array>
#include "odeSolve.h"

#define dt 2.0e-5
#define SAVEFREQ 1
#define STEPFREQ 1
#define maxStep 100000

using namespace std;

vector<double> u,v,r, x,y,thetaR;
vector<double> tt, forceX;

int main(int argv, char **argc) {

	const modelParams params;

	u.reserve(maxStep); v.reserve(maxStep); r.reserve(maxStep);
	x.reserve(maxStep); y.reserve(maxStep); thetaR.reserve(maxStep);
	tt.reserve(maxStep); forceX.reserve(maxStep);

	double p[2] = {0.0, 0.0};

	//u.push_back(distribU(genA)), v.push_back(distribU(genA)); // random initial linear velocities
	u.push_back(10), v.push_back(0.0); // random initial linear velocities
	r.push_back(0.0);
	x.push_back(p[0]), y.push_back(p[1]), thetaR.push_back(0.0);
	tt.push_back(0.0);

	double actions[2];
	actions[0] = -20.0; // Thruster Left
	actions[1] = 20.0; // Thruster Right

	for (int step=0; step<maxStep; ++step){

		forceX[step] = actions[0] + actions[1];
		const double forceY = 0.0;

		// Assume no other forces for the time being
		const double thrustL = actions[0];
		const double thrustR = actions[1];
		const double torque = 0.5*params.l*(thrustR - thrustL);

		const double uN[3] = {u.back(), v.back(), r.back()};
		double uNp1[3] = {0.0};

		odeSolve(uN, params, dt, forceX.back(), forceY, torque, uNp1);
		u.push_back(uNp1[0]), v.push_back(uNp1[1]), r.push_back(uNp1[2]);

		const double xN[3] = {x.back(), y.back(), thetaR.back()};
		double xNp1[3] = {0.0};
		trajectory(xN, uN, uNp1, dt, xNp1);
		x.push_back(xNp1[0]), y.push_back(xNp1[1]), thetaR.push_back(xNp1[2]);

		p[0] = xNp1[0];
		p[1] = xNp1[1];

		tt.push_back(dt*(step+1));
	}

	FILE *temp = fopen("traj.txt", "w");
	fprintf(temp, "time \t forceX \t forceY \t u \t v \t r \t x \t y \t theta\n");
	for(unsigned int i=0; i<tt.size(); i++)  fprintf(temp, "%f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f\n", tt[i], forceX[i], 0.0, u[i], v[i], r[i], x[i], y[i], thetaR[i]);
	fclose(temp);

}
