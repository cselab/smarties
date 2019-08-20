#include "odeSolve.h"

void getDerivs(const modelParams p, const double Fx, const double Fy, const double Tau, const double u, const double v, const double r, const double vDot, const double rDot, double retVal[3])
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
void odeSolve(const double uNot[3], const modelParams params, const int N, const double *tt, 
		const double *fX, const double *fY, const double *torque,
		double *u, double *v, double *r) 
{

	double *uDot, *vDot, *rDot;
	uDot = new double[N]();	
	vDot = new double[N]();	
	rDot = new double[N]();	

	// Let's say start from rest:
	uDot[0] = vDot[0] = rDot[0] = 0.0;
	u[0]=uNot[0]; v[0]=uNot[1]; r[0]=uNot[2];

	for(int i=0; i<(N-1); i++){

		const double dt = tt[i+1] - tt[i];

		//RK4
		double stage1[3]={0}, stage2[3]={0}, stage3[3]={0}, stage4[3]={0};
		getDerivs(params, fX[i], fY[i], torque[i], u[i], v[i], r[i], vDot[i], rDot[i], stage1);
		getDerivs(params, fX[i], fY[i], torque[i], u[i]+(0.5*dt*stage1[0]), v[i]+(0.5*dt*stage1[1]), r[i]+(0.5*dt*stage1[2]), vDot[i], rDot[i], stage2);
		getDerivs(params, fX[i], fY[i], torque[i], u[i]+(0.5*dt*stage2[0]), v[i]+(0.5*dt*stage2[1]), r[i]+(0.5*dt*stage2[2]), vDot[i], rDot[i], stage3);
		getDerivs(params, fX[i], fY[i], torque[i], u[i]+(0.5*dt*stage3[0]), v[i]+(0.5*dt*stage3[1]), r[i]+(0.5*dt*stage3[2]), vDot[i], rDot[i], stage4);

		uDot[i+1] = (stage1[0] + 2*stage2[0] + 2*stage3[0] + stage4[0]);
		vDot[i+1] = (stage1[1] + 2*stage2[1] + 2*stage3[1] + stage4[1]);
		rDot[i+1] = (stage1[2] + 2*stage2[2] + 2*stage3[2] + stage4[2]);

		u[i+1] = u[i] + (dt/6.0)*uDot[i+1];
		v[i+1] = v[i] + (dt/6.0)*vDot[i+1];
		r[i+1] = r[i] + (dt/6.0)*rDot[i+1];
	}

	delete[] uDot; delete[] vDot; delete[] rDot;
}

//////////////////////////////////////////////////////////////////////////////////////////
void trajectory(const int N, const double *tt, const double *u, const double *v, const double *r,
		double *x, double *y, double *theta) 
{
	// Let's say start from rest:
	x[0] = y[0] = theta[0] = 0.0;
	for(int i=0; i<(N-1); i++){
		const double dt = tt[i+1] - tt[i];

		x[i+1] = x[i] 		+ (dt/2.0)*(u[i] + u[i+1]);
		y[i+1] = y[i] 		+ (dt/2.0)*(v[i] + v[i+1]);
		theta[i+1] = theta[i] 	+ (dt/2.0)*(r[i] + r[i+1]);
	}
}
