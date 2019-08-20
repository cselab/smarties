#pragma once
#include<cmath>
#include<cstdio>

struct modelParams
{
        double m = 280;
        double Izz = 300;
        double l = 1.83;
        double Xu = 86.45;
        double Xuu = 0;
        double Yv = 300;
        double Nr = 500;
        double Nv = -250;
        double Yr = -80;
        double Nu = 20;
        double XuDot = -30;
        double YvDot = -40;
        double NrDot = -90;
        double NvDot = -50;
        double YrDot = -50;

	const double M[3][3] = { {m-XuDot, 0, 0},
			{0, m-YvDot, -YrDot},
			{0, -NvDot, Izz-NrDot} 	};

	const double determinant = std::abs( M[0][0]* (M[1][1]*M[2][2] - M[1][2]*M[2][1]) ); // others are zero
	double invM[3][3];

	const double D[3][3] = {{Xu, 0, 0}, 
				{0,  Yv, Yr},
				{0,  Nv, Nr} };

	modelParams(){

		invM[0][0] =  (M[1][1]*M[2][2] - M[2][1]*M[1][2]) /determinant;
		invM[0][1] = -(M[1][0]*M[2][2] - M[2][0]*M[1][2]) /determinant;
		invM[0][2] =  (M[1][0]*M[2][1] - M[2][0]*M[1][1]) /determinant;

		invM[1][0] = -(M[0][1]*M[2][2] - M[2][1]*M[0][2]) /determinant;
		invM[1][1] =  (M[0][0]*M[2][2] - M[2][0]*M[0][2]) /determinant;
		invM[1][2] = -(M[0][0]*M[2][1] - M[2][0]*M[0][1]) /determinant;

		invM[2][0] =  (M[0][1]*M[1][2] - M[1][1]*M[0][2]) /determinant;
		invM[2][1] = -(M[0][0]*M[1][2] - M[1][0]*M[0][2]) /determinant;
		invM[2][2] =  (M[0][0]*M[1][1] - M[1][0]*M[0][1]) /determinant;

		/*printf("invM = %20.18f \t %20.18f \t %20.18f \n %20.18f \t %20.18f \t %20.18f \n %20.18f \t %20.18f \t %20.18f \n", 
				invM[0][0],invM[0][1],invM[0][2], 
				invM[1][0],invM[1][1],invM[1][2],
				invM[2][0],invM[2][1],invM[2][2]);
		abort();*/
	}


};

void getDerivs(const modelParams p, const double Fx, const double Fy, const double Tau, const double u, const double v, const double r, double retVal[3]);

void odeSolve(const double uVec[3], const modelParams params, const double dt, const double fX, const double fY, const double torque, double uOut[3]);

void trajectory(const double xVec[3], const double uN[3], const double uNp1[3], const double dt, double xOut[3]);
