#include "cmaes_interface.h"
#include "Communicator.h"
#include <math.h>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <sstream>

#include "cmaes_learn.h"
#include "fitfun.h"


void random_action( cmaes_t* const evo, std::mt19937 gen ){

  std::uniform_real_distribution<double> act0dist(.02,0.1);
  std::uniform_real_distribution<double> act1dist(.02,0.1);
  std::uniform_real_distribution<double> act2dist(.2,.8);
  std::uniform_real_distribution<double> act3dist(.2,.8);


  evo->sp.ccov1   = act0dist(gen);
  evo->sp.ccovmu  = act1dist(gen);
  evo->sp.ccumcov = act2dist(gen);
  evo->sp.cs      = act3dist(gen);
  update_damps( evo );

}

void write_cmaes_perf::write( const int thrid ){
  char filename[256];
  sprintf(filename, "cma_perf_%02d.dat", thrid);
  FILE *fp = fopen(filename, "a");
  fprintf(fp, "func_id dim nstep dist feval\n");
  fclose(fp);
}

void write_cmaes_perf::write( cmaes_t* const evo, const int thrid, const int func_dim, int func_id, int step, const double final_dist, double ffinal ){
  //print to file dim, function ID, nsteps, distance from opt, function value
  char filename[256];
  FILE *fp;

  sprintf(filename, "cma_perf_%02d.dat", thrid);
  fp = fopen(filename, "a");
  fprintf(fp, "%d %d %d %e %e\n", func_id, func_dim, step, final_dist, ffinal);
  fclose(fp);


  const double *xbever = cmaes_GetPtr(evo, "xbestever");

  sprintf(filename, "cma_xbever_%02d.dat", thrid);
  fp = fopen(filename, "a");
  for(int i=0; i<func_dim; i++)
    fprintf(fp,"%lf  ",xbever[i]);
  fprintf(fp,"\n");

  fclose(fp);
}



bool evaluate_and_update( cmaes_t* const evo, double* const*  pop, double *arFunvals, int* const info  ){

  int lambda = evo->sp.lambda;
  int func_dim = evo->sp.N;

  for (int i = 0; i < lambda; i++)
    fitfun(pop[i], func_dim, &arFunvals[i], info);

  cmaes_UpdateDistribution(evo, arFunvals);

  if(evo->isStuck == 1) return true;

  return false;

}

bool check_for_nan_inf(cmaes_t* const evo, double* const* pop ){

  int lambda   = evo->sp.lambda;
  int func_dim = evo->sp.N;

  bool foundnan = false;

  for (int i = 0; i < lambda; ++i)
    for (int j = 0; j < func_dim; j++)
      if (std::isnan(pop[i][j]) || std::isinf(pop[i][j]))
        foundnan = true;

  if(foundnan){
    fprintf(stderr, "It was nan all along!!!\n");
    evo->isStuck = 1;
    return true;
  }

  return false;
}

void Action::update( cmaes_t* const evo, double **arFunvals )
{
  // smarties just set a vector of unbounded actions
  // all arguments of cma are bounded: map actions

  if (dim>0) { //rank 1 covariance update
    // const double shift0 = zeroOneInv(default_0);
    // evo->sp.ccov1  = zeroOneMap(shift0+data[0]);
    evo->sp.ccov1  = default_0*zeroInfMap(data[0]);
    //std::cout << evo->sp.ccov1 << " " << default_0 << " " << data[0] << std::endl;
  }

  if (dim>1) { //rank mu covariance update
    // const double shift0 = zeroOneInv(default_1);
    // evo->sp.ccovmu  = zeroOneMap(shift0+data[1]);
    evo->sp.ccovmu = default_1*zeroInfMap(data[1]);
    //std::cout << evo->sp.ccovmu << " " << default_1 << " " << data[1] << std::endl;
  }

  if (dim>2) { //path update c_c
    // const double shift0 = zeroOneInv(default_2);
    // evo->sp.ccumcov  = zeroOneMap(shift0+data[2]);
    evo->sp.ccumcov = default_2*zeroInfMap(data[2]);
    //std::cout << evo->sp.ccumcov << " " << default_2 << " " << data[2] << std::endl;
  }

  if (dim>3) { //step size control c_sigmai
  //   const double shift0 = zeroOneInv(default_3);
  //   evo->sp.cs  = zeroOneMap(shift0+data[3]);
    evo->sp.cs      = default_3*zeroInfMap(data[3]);
    //std::cout << evo->sp.cs << " " << default_3 << " " << data[3] << std::endl;
  }

  if (dim>5) { //pop size ratio
    lambda_frac = data[5];
    const double default_5   = 4+std::floor(3*std::log(func_dim));
    lambda = std::floor( default_5*zeroInfMap(data[5]) );
    lambda   = lambda < 4 ? 4 : lambda;
    *arFunvals   = cmaes_ChangePopSize(evo, lambda);
  }

  update_damps( evo );
  if (dim>4) { //rank mu covariance update
    const double default_4 = evo->sp.damps;
    // const double shift0 = zeroInfInv(default_4);
    // evo->sp.damps  = zeroInfMap(shift0+data[4]);
    evo->sp.damps   = default_4*zeroInfMap(data[4]);
  }
}

void dump_curgen( double* const* pop, double *arFunvals, int step, int lambda, int func_dim ){
  char filename[256];
  sprintf(filename, "curgen_db_%03d.txt", step);
  FILE *fp = fopen(filename, "w");

  for (int i = 0; i < lambda; i++){
    for (int j = 0; j < func_dim; j++)
      fprintf(fp, "%.6le ", pop[i][j]);
    fprintf(fp, "%.6le\n", arFunvals[i]);
  }
  fclose(fp);
}


void print_best_ever( cmaes_t* const evo,  int step ){
  int func_dim = evo->sp.N;
  const double *xbever = cmaes_GetPtr(evo, "xbestever");
  double fbever = cmaes_Get(evo, "fbestever");

  printf("BEST @ %5d: ", step);
  for (int i = 0; i < func_dim; i++)
    printf("%25.16lf ", xbever[i]);
  printf("%25.16lf\n", fbever);
}

void update_damps( cmaes_t* const evo )
{
  int lambda = evo->sp.lambda;
  int N = evo->sp.N;
  double dampFac = std::min(evo->sp.stopMaxIter,evo->sp.stopMaxFunEvals/lambda);
  evo->sp.damps = //basic factor:
                  (1 + 2*std::max(0., std::sqrt((evo->sp.mueff-1.)/(N+1.)) -1) )
                  // anti short runs:
                  * std::max(0.3, 1. -N/(1e-6+dampFac) ) + evo->sp.cs;
}

bool resample( cmaes_t* const evo, double* const* pop, double* const lower_bound, double* const upper_bound){
  int lambda = evo->sp.lambda;
  int func_dim = evo->sp.N;
  int safety = 0;
  for (int i = 0; i < lambda; ++i){
    while ( !is_feasible(pop[i],lower_bound,upper_bound,func_dim) && safety++ < 1e4){
      cmaes_ReSampleSingle(evo, i);
      if(evo->isStuck == 1) return true;
    }
  }

  return false;
}


int is_feasible(double* const pop, double* const lower_bound, double* const upper_bound, int dim)
{
  int good;
  for (int i = 0; i < dim; i++) {
    if (std::isnan(pop[i]) || std::isinf(pop[i])){
      printf("Sampled nan: FU cmaes \n");
      abort();
    }

    good = (lower_bound[i] <= pop[i]) && (pop[i] <= upper_bound[i]);
    if (!good) return 0;
  }
  return 1;
}

void State::update_state(cmaes_t*const evo, double& oldFm, double*oldXmean)
{
  double* xMean = cmaes_GetNew(evo, "xmean");

  double xProgress = 0.;
  for(int i=0; i<func_dim; i++) xProgress += std::pow(xMean[i]-oldXmean[i],2);

  const double fmedian = cmaes_Get(evo, "fmedian");
  const double progress = (fmedian-oldFm)/(std::fabs(fmedian)+std::fabs(oldFm));
  const double ratio1 = evo->mindiagC/(evo->maxdiagC+1e-16);
  const double ratio2 = evo->minEW   /(evo->maxEW   +1e-16);
  //distinction makes sense if function is rotated

  #if 0
  double modality;
  if(lambda > func_dim)
  { //plane equation overfitted in sample points

    Eigen::VectorXd y(lambda);

    for(int i=0; i<lambda; i++){ y(i) = evo->rgFuncValue[i]; }

    Eigen::MatrixXd X(lambda,func_dim);

    for(int i=0; i<lambda; i++){
      for(int j=0; j<func_dim; j++){
        X(i,j) = evo->rgrgx[i][j];
      }
    }

    //solve least squares plane fitting using QR-partition
    Eigen::VectorXd d = X * (X.colPivHouseholderQr().solve(y)) - y;
    modality = d.norm();
  }
  else { modality = -1; }
  #endif

  //prevent nans/infs from infecting the delicate snowflake that is the RL code
  if (std::isnan(ratio1) || std::isinf(ratio1))
  { perror("Ratio1 is nan, FU CMAES \n"); abort(); }
  if (std::isnan(ratio2) || std::isinf(ratio2))
  { perror("Ratio2 is nan, FU CMAES \n"); abort(); }
  if (std::isnan(xProgress) || std::isinf(xProgress))
  { perror("xProgress is nan, FU CMAES \n"); abort(); }
  if (std::isnan(progress) || std::isinf(progress))
  { perror("fProgress is nan, FU CMAES \n"); abort(); }
  if (std::isnan(evo->trace) || std::isinf(evo->trace))
  { perror("evo->trace is nan, FU CMAES \n"); abort(); }

  data[0] = ratio1;
  data[1] = ratio2;
  data[2] = sqrt(xProgress);
  data[3] = progress;
  data[4] = (double)func_dim;
  data[5] = evo->trace;

  //advance:
  oldFm = fmedian;
  for (int i=0; i<func_dim; i++) oldXmean[i] = xMean[i];
  free(xMean);
}
