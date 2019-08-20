#include "odeSolve.h"

void getDerivs(const modelParams p, const double Fx, const double Fy, const double Tau, const double u, const double v, const double r, double retVal[3])
{

	const double C[3][3] = {{0,  0,   -p.m*v +  p.YvDot*v + (p.YrDot + p.NvDot)*r/2.0 },
				{0,  0,   p.m*u -p.XuDot*u},
				{p.m*v -(p.YvDot*v + (p.YrDot + p.NvDot)*r/2),  -p.m*u + p.XuDot*u,  0} };


	const double temp1[3] = {	u*(C[0][0]+p.D[0][0]) + v*(C[0][1]+p.D[0][1]) + r*(C[0][2]+p.D[0][2]) ,
					u*(C[1][0]+p.D[1][0]) + v*(C[1][1]+p.D[1][1]) + r*(C[1][2]+p.D[1][2]),
					u*(C[2][0]+p.D[2][0]) + v*(C[2][1]+p.D[2][1]) + r*(C[2][2]+p.D[2][2]) };

	const double temp2[3] = {	p.invM[0][0]*temp1[0] + p.invM[0][1]*temp1[1] + p.invM[0][2]*temp1[2],
					p.invM[1][0]*temp1[0] + p.invM[1][1]*temp1[1] + p.invM[1][2]*temp1[2],
					p.invM[2][0]*temp1[0] + p.invM[2][1]*temp1[1] + p.invM[2][2]*temp1[2] };

	retVal[0] = Fx  - temp2[0];
	retVal[1] = Fy  - temp2[1];
	retVal[2] = Tau - temp2[2];

	/*retVal[0] = (Fx + r*(p.YvDot*v + p.m*v - r*(p.NvDot + p.YrDot)/2) - u*(p.Xu + p.Xuu*u)) / (p.m-p.XuDot);
	  retVal[1] = (Fy - r*(p.XuDot*u + p.m*u) - p.Yv*v + p.YrDot*rDot - p.Yr*r) / (p.m-p.YvDot);
	  retVal[2] = (Tau + p.NvDot*vDot - p.Nv*v - u*(p.YvDot*v + p.m*v + r*(p.NvDot+p.YrDot)/2) + p.Nr*r + v*(p.XuDot*u + p.m*u)) / (p.Izz-p.NrDot);*/
}


//////////////////////////////////////////////////////////////////////////////////////////
void odeSolve(const double uVec[3], const modelParams params, const double dt, 
		const double fX, const double fY, const double torque, double uOut[3]) 
{
	const double u=uVec[0], v=uVec[1], r=uVec[2];
	double uDotOut[3] = {0.0};

	//RK4
	double stage1[3]={0}, stage2[3]={0}, stage3[3]={0}, stage4[3]={0};
	getDerivs(params, fX, fY, torque, u, v, r, stage1);
	getDerivs(params, fX, fY, torque, u+(0.5*dt*stage1[0]), v+(0.5*dt*stage1[1]), r+(0.5*dt*stage1[2]), stage2);
	getDerivs(params, fX, fY, torque, u+(0.5*dt*stage2[0]), v+(0.5*dt*stage2[1]), r+(0.5*dt*stage2[2]), stage3);
	getDerivs(params, fX, fY, torque, u+(0.5*dt*stage3[0]), v+(0.5*dt*stage3[1]), r+(0.5*dt*stage3[2]), stage4);

	uDotOut[0] = (stage1[0] + 2*stage2[0] + 2*stage3[0] + stage4[0]);
	uDotOut[1] = (stage1[1] + 2*stage2[1] + 2*stage3[1] + stage4[1]);
	uDotOut[2] = (stage1[2] + 2*stage2[2] + 2*stage3[2] + stage4[2]);

	uOut[0] = u + (dt/6.0)*uDotOut[0];
	uOut[1] = v + (dt/6.0)*uDotOut[1];
	uOut[2] = r + (dt/6.0)*uDotOut[2];
}

//////////////////////////////////////////////////////////////////////////////////////////
void trajectory(const double xVec[3], const double uN[3], const double uNp1[3], const double dt, double xOut[3]) 
{
	xOut[0] = xVec[0] + (dt/2.0)*(uN[0] + uNp1[0]);
	xOut[1] = xVec[1] + (dt/2.0)*(uN[1] + uNp1[1]);
	xOut[2] = xVec[2] + (dt/2.0)*(uN[2] + uNp1[2]);
}
