#include<iostream>
#include<cmath>
#include"odeSolve.h"
#include<stdio.h>
#include<random>
using namespace std;


/////////////////////////////
int main()
{
	std::mt19937 gen;
	std::uniform_real_distribution<> dist(0, 10);

	const modelParams params;
	const double uNot[3] = {-10,0,0};
	const double tStart = 0, tEnd = 2;
	const int N = 100000;

	double *u, *v, *r;
	double *x, *y, *theta;
	double *tt, *thrustL, *thrustR, *torque;
	double *fXext, *fYext, *forceX, *forceY, *torqueExt;

	u = new double[N](); v = new double[N](); r = new double[N]();
	x = new double[N](); y = new double[N](); theta = new double[N]();
	tt = new double[N](); thrustL = new double[N](); thrustR = new double[N](); torque = new double[N]();
	fXext = new double[N](); fYext = new double[N]();
	forceX = new double[N](); forceY = new double[N]();

	const double dt = (tEnd-tStart)/(N-1);
	//const double angFactor = 2*M_PI/(tEnd - tStart);

	for (int i=0; i<N; i++){
		tt[i] = tStart + i*dt;

		thrustL[i] = 20.0; //cos(tt[i]*angFactor/4.0);
		thrustR[i] = -20.0;
		torque[i] = (0.5*params.l)*(thrustR[i] - thrustL[i]);
		fXext[i] = 0.0;//*cos(tt[i]*angFactor/4.0);
		fYext[i] = 0.0; //dist(gen)*sin(tt[i]*angFactor);

		forceX[i] = thrustL[i] + thrustR[i] + fXext[i];
		forceY[i] = fYext[i];
	}

	odeSolve(uNot, params, N, tt, forceX, forceY, torque, u, v, r);
	trajectory(N, tt, u, v, r, x, y, theta);

	FILE *temp = fopen("trajectory.txt", "w");
	fprintf(temp, "time \t forceX \t forceY \t torque \t u \t v \t r \t x \t y \t theta\n");
	for(int i=0; i<N; i++)	fprintf(temp, "%f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f\n", tt[i], forceX[i], forceY[i], torque[i], u[i], v[i], r[i], x[i], y[i], theta[i]);
	fclose(temp);

	return 0;
}
