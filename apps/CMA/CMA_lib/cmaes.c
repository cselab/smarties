/* --------------------------------------------------------- */
/* --- File: cmaes.c  -------- Author: Nikolaus Hansen   --- */
/* --------------------------------------------------------- */
/*
   CMA-ES for non-linear function minimization.

   Copyright 1996, 2003, 2007, 2013 Nikolaus Hansen
   e-mail: hansen .AT. lri.fr

SOURCE:
https://github.com/cma-es/c-cma-es
https://github.com/cma-es/c-cma-es/blob/master/src/cmaes.c

LICENSE: this library is free/open software and may be used
either under the

Apache License 2.0

or under the

GNU Lesser General Public License 2.1 or later

whichever suits best.

See also the LICENSE file
https://github.com/cma-es/c-cma-es/blob/master/LICENSE
*/

/* --- Changes : ---
   03/03/21: argument const double *rgFunVal of
   cmaes_ReestimateDistribution() was treated incorrectly.
   03/03/29: restart via cmaes_resume_distribution() implemented.
   03/03/30: Always max std dev / largest axis is printed first.
   03/08/30: Damping is adjusted for large mueff.
   03/10/30: Damping is adjusted for large mueff always.
   04/04/22: Cumulation time and damping for step size adjusted.
   No iniphase but conditional update of pc.
   05/03/15: in ccov-setting mucov replaced by mueff.
   05/10/05: revise comment on resampling in example.c
   05/10/13: output of "coorstddev" changed from sigma * C[i][i]
   to correct sigma * sqrt(C[i][i]).
   05/11/09: Numerical problems are not anymore handled by increasing
   sigma, but lead to satisfy a stopping criterion in
   cmaes_Test().
   05/11/09: Update of eigensystem and test for numerical problems
   moved right before sampling.
   06/02/24: Non-ansi array definitions replaced (thanks to Marc
   Toussaint).
   06/02/25: Overflow in time measurement for runs longer than
   2100 seconds. This could lead to stalling the
   covariance matrix update for long periods.
   Time measurement completely rewritten.
   06/02/26: Included population size lambda as parameter to
   cmaes_init (thanks to MT).
   06/02/26: Allow no initial reading/writing of parameters via
   "non" and "writeonly" keywords for input parameter
   filename in cmaes_init.
   06/02/27: Optimized code regarding time spent in updating the
   covariance matrix in function Adapt_C2().
   07/08/03: clean up and implementation of an exhaustive test
   of the eigendecomposition (via #ifdef for now)
   07/08/04: writing of output improved
   07/08/xx: termination criteria revised and more added,
   damp replaced by damps=damp*cs, documentation improved.
   Interface significantly changed, evaluateSample function
   and therefore the function pointer argument removed.
   Renaming of functions in accordance with Java code.
   Clean up of parameter names, mainly in accordance with
   Matlab conventions. Most termination criteria can be
   changed online now. Many more small changes, but not in
   the core procedure.
   07/10/29: ReSampleSingle() got a better interface. ReSampleSingle()
   is now ReSampleSingle_old only for backward
   compatibility. Also fixed incorrect documentation. The new
   function SampleSingleInto() has an interface similar to
   the old ReSampleSingle(), but is not really necessary.
   07/11/20: bug: stopMaxIter did not translate into the correct default
   value but into -1 as default. This lead to a too large
   damps and the termination test became true from the first
   iteration. (Thanks to Michael Calonder)
   07/11/20: new default stopTolFunHist = 1e-13;  (instead of zero)
   08/09/26: initial diagonal covariance matrix in code, but not
   yet in interface
   08/09/27: diagonalCovarianceMatrix option in initials.par provided
   08/10/17: uncertainty handling implemented in example3.c.
   PerturbSolutionInto() provides the optional small
   perturbations before reevaluation.
   10/10/16: TestForTermination changed such that diagonalCovarianceMatrix
   option now yields linear time behavior
   12/05/28: random seed > 2e9 prohibited to avoid an infinite loop on 32bit systems
   12/10/21: input parameter file values "no", "none" now work as "non".
   12/10/xx: tentative implementation of cmaes_Optimize
   12/10/xx: some small changes with char * mainly to prevent warnings in C++
   12/10/xx: added some string convenience functions isNoneStr, new_string, assign_string
   13/01/03: rename files example?, initials.par, signals.par
   14/04/29: removed bug, au = t->al[...], from the (new) boundary handling
   code (thanks to Emmanuel Benazera for the hint)
   14/06/16: test of un-initialized version number removed (thanks to Paul Zimmermann)
   14/10/18: splitted cmaes_init() into cmaes_init_para() and cmaes_init_final(),
such that parameters can be set also comparatively safely within the
code, namely after calling cmaes_init_para(). The order of parameters
<<<<<<< HEAD
  of readpara_init() has changed to be the same as for cmaes_init().
14/11/25  fix warnings from Microsoft C compiler (sherm)
  =======
  of readpara_init() has changed to be the same as for cmaes_init().
  14/11/26: renamed exported symbols so they begin with a cmaes_prefix.
  >>>>>>> bd5aeb2a21fbc437f1b322610022f7cb1ba6a9db

  Wish List
o make signals_filename part of cmaes_t using assign_string()

  o as writing time is measure for all files at once, the display
  cannot be independently written to a file via signals.par, while
  this would be desirable.

  o clean up sorting of eigenvalues and vectors which is done repeatedly.

  o either use cmaes_Get() in cmaes_WriteToFilePtr(): revise the
                            cmaes_write that all keywords available with get and getptr are
                                                 recognized. Also revise the keywords, keeping backward
                                                              compatibility. (not only) for this it would be useful to find a
                                                                         way how cmaes_Get() signals an unrecognized keyword. For GetPtr
                                                                                it can return NULL.

                                                                                      o or break cmaes_Get() into single getter functions, being a nicer
                                                                                      interface, and compile instead of runtime error, and faster. For
                                                                                      file signals.par it does not help.

                                                                                      o writing data depending on timing in a smarter way, e.g. using 10%
                                                                                      of all time. First find out whether clock() is useful for measuring
                                                                                      disc writing time and then cmaes_timings_t class can be utilized.
                                                                                      For very large dimension the default of 1 seconds waiting might
                                                                                      be too small.

o allow modification of best solution depending on delivered f(xmean)

  o re-write input and output procedures
  */

  /* Prevent Microsoft compiler from complaining that common library functions
     like strncpy(), ctime(), sprintf(), fopen(), fscanf(), etc. "may be unsafe".
     This must come before the first #include.
     */
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <math.h>   /* sqrt() */
#include <stddef.h> /* size_t */
#include <stdlib.h> /* NULL, free */
#include <string.h> /* strlen() */
#include <stdio.h>  /* sprintf(), NULL? */
#include "cmaes_interface.h" /* <time.h> via cmaes.h */

  /* --------------------------------------------------------- */
  /* ------------------- Declarations ------------------------ */
  /* --------------------------------------------------------- */

  /* ------------------- External Visibly -------------------- */

  /* see cmaes_interface.h for those, not listed here */

  long   cmaes_random_init(cmaes_random_t *, long unsigned seed /* 0==clock */);
  void   cmaes_random_exit(cmaes_random_t *);
  double cmaes_random_Gauss(cmaes_random_t *); /* (0,1)-normally distributed */
  double cmaes_random_Uniform(cmaes_random_t *);
  long   cmaes_random_Start(cmaes_random_t *, long unsigned seed /* 0==1 */);

  void   cmaes_timings_init(cmaes_timings_t *timing);
  void   cmaes_timings_start(cmaes_timings_t *timing); /* fields totaltime and tictoctime */
  double cmaes_timings_update(cmaes_timings_t *timing);
  void   cmaes_timings_tic(cmaes_timings_t *timing);
  double cmaes_timings_toc(cmaes_timings_t *timing);

  void cmaes_readpara_init (cmaes_readpara_t * const, int dim, const double * const xstart,
      const double * const sigma, int seed, int lambda,
      const char * filename);
void cmaes_readpara_exit(cmaes_readpara_t * const);
void cmaes_readpara_ReadFromFile(cmaes_readpara_t * const, const char *szFileName);
void cmaes_readpara_SupplementDefaults(cmaes_readpara_t * const);
void cmaes_readpara_SetWeights(cmaes_readpara_t * const, const char * mode);
void cmaes_readpara_WriteToFile(cmaes_readpara_t * const, const char *filenamedest);

const double * cmaes_Optimize( cmaes_t * const, double(*pFun)(double const *, int dim),  long iterations);
double const * cmaes_SetMean(cmaes_t * const, const double *xmean);
/*double * cmaes_PerturbSolutionInto(cmaes_t *t, double *xout,
  double const *xin, double eps);*/

void cmaes_WriteToFileAW(cmaes_t * const t, const char *key, const char *name,
    const char * append);
void cmaes_WriteToFilePtr(cmaes_t *, const char *key, FILE *fp);
void cmaes_ReadFromFilePtr(cmaes_t *, FILE *fp);
void cmaes_FATAL(char const *s1, char const *s2, char const *s3, char const *s4);


/* ------------------- Locally visibly ----------------------- */

static char * getTimeStr(void);
static void TestMinStdDevs( cmaes_t *);
/* static void WriteMaxErrorInfo( cmaes_t *); */

static void Eigen( int N,  double ** const C, double * const diag, double ** const Q, double *rgtmp);
static int  Check_Eigen( int N,  double ** const C, double * const diag, double ** const Q);
static void QLalgo2 (int n, double * const d, double * const e, double ** const V);
static void Householder2(int n, double ** const V, double * const d, double * const e);
static void Adapt_C2(cmaes_t * const t, int hsig);

static void FATAL(char const *sz1, char const *s2,
    char const *s3, char const *s4);
static void ERRORMESSAGE(char const *sz1, char const *s2,
    char const *s3, char const *s4);
static int isNoneStr(const char * filename);
static void   Sorted_index( const double *rgFunVal, int *index, int n);
static int    SignOfDiff( const void *d1, const void * d2);
static double douSquare(double);
static double rgdouMax( const double *rgd, int len);
static double rgdouMin( const double *rgd, int len);
static double douMax( double d1, double d2);
static double douMin( double d1, double d2);
static int    intMin( int i, int j);
static int    MaxIdx( const double *rgd, int len);
static int    MinIdx( const double *rgd, int len);
static double myhypot(double a, double b);
static double * new_double( int n);
static void * new_void( int n, size_t size);
static char * new_string( const char *);
static void assign_string( char **, const char*);

static const char * c_cmaes_version = "3.20.00.beta";

/* --------------------------------------------------------- */
/* ---------------- Functions: cmaes_t --------------------- */
/* --------------------------------------------------------- */

static char *
getTimeStr(void) {
  time_t tm = time(NULL);
  static char s[33];

  /* get time */
  strncpy(s, ctime(&tm), 24); /* TODO: hopefully we read something useful */
  s[24] = '\0'; /* cut the \n */
  return s;
}

  char *
cmaes_SayHello(cmaes_t * const t)
{
  /* write initial message */
  sprintf(t->sOutString,
      "(%d,%d)-CMA-ES(mu_eff=%.1f), Ver=\"%s\", dimension=%d, diagonalIterations=%ld, randomSeed=%d (%s)",
      t->sp.mu, t->sp.lambda, t->sp.mueff, t->version, t->sp.N, (long)t->sp.diagonalCov, t->sp.seed, getTimeStr());

  return t->sOutString;
}

/* --------------------------------------------------------- */
/* --------------------------------------------------------- */
  void
cmaes_init_para(cmaes_t * const t, /* "this" */
    int dimension,
    double * const inxstart,
    double * const inrgstddev, /* initial stds */
    long int inseed,
    int lambda,
    const char *input_parameter_filename)
{
  t->version = c_cmaes_version;
  cmaes_readpara_init(&t->sp, dimension, inxstart, inrgstddev, inseed,
      lambda, input_parameter_filename);
}

  double *
cmaes_init_final(cmaes_t * const t /* "this" */)
  /*
   * */
{
  int i, j, N;
  double trace;

  if (t->version == NULL) {
    ERRORMESSAGE("cmaes_init_final called (probably) without calling cmaes_init_para first",
        ", which will likely lead to unexpected results",0,0);
    printf("Error: cmaes_init_final called (probably) without calling cmaes_init_para first\n");
  }
  if (strcmp(c_cmaes_version, t->version) != 0) {
    ERRORMESSAGE("cmaes_init_final called twice, which will lead to a memory leak",
        "; use cmaes_exit() first",0,0);
    printf("Error: cmaes_init_final called twice, which will lead to a memory leak; use cmaes_exit first\n");
  }
  /* assign_string(&t->signalsFilename, "cmaes_signals.par"); */

  if (!t->sp.flgsupplemented) {
    cmaes_readpara_SupplementDefaults(&t->sp);
    if (!isNoneStr(t->sp.filename)) /* TODO: should this be done in readpara_SupplementDefaults? */
      cmaes_readpara_WriteToFile(&t->sp, "actparcmaes.par");
  }

  t->sp.seed = cmaes_random_init( &t->rand, (long unsigned int) t->sp.seed);

  N = t->sp.N; /* for convenience */
  if (N<1) {
    printf("Dimensionality abort\n");
    abort();
  }
  /* initialization  */
  for (i = 0, trace = 0.; i < N; ++i)
    trace += t->sp.rgInitialStds[i]*t->sp.rgInitialStds[i];
  t->sigma = sqrt(trace/N); /* t->sp.mueff/(0.2*t->sp.mueff+sqrt(N)) * sqrt(trace/N); */

  t->chiN = sqrt((double) N) * (1. - 1./(4.*N) + 1./(21.*N*N));
  t->flgEigensysIsUptodate = 1;
  t->flgCheckEigen = 0;
  t->genOfEigensysUpdate = 0;
  cmaes_timings_init(&t->eigenTimings);
  t->flgIniphase = 0; /* do not use iniphase, hsig does the job now */
  t->flgresumedone = 0;
  t->flgStop = 0;

  t->dMaxSignifKond = 9e+12; /* not sure whether this is really save, 100 does not work well enough */

  t->gen = 0;
  t->countevals = 0;
  t->state = 0;
  t->dLastMinEWgroesserNull = 1.0;
  t->printtime = t->writetime = t->firstwritetime = t->firstprinttime = 0;
  t->rgpc = new_double(N);
  t->rgps = new_double(N);
  t->rgdTmp = new_double(N+1);
  t->rgBDz = new_double(N);
  t->rgxmean = new_double(N+1);
  t->rgxold = new_double(N+1);
  t->rgxbestever = new_double(N+2);
  t->rgout = new_double(N+1);
  t->rgD = new_double(N);
  t->C = (double**)new_void(N, sizeof(double*));
  t->B = (double**)new_void(N, sizeof(double*));
  for (i = 0; i < N; ++i) {
    t->C[i] = new_double(i+1);
    t->B[i] = new_double(N);
  }
  //GN: changed size of array: made independent of lambda
  t->arFuncValueHist = new_double(10+(int)ceil(30.*N));

  ///////////////////////////////////////////////////
  t->publicFitness = new_double(t->sp.lambda);
  t->rgFuncValue = new_double(t->sp.lambda);

  t->index = (int *) new_void(t->sp.lambda, sizeof(int));
  for (i = 0; i < t->sp.lambda; ++i)
    t->index[i] = i; /* should not be necessary */
  t->rgrgx = (double **)new_void(t->sp.lambda, sizeof(double*));
  for (i = 0; i < t->sp.lambda; ++i)
    t->rgrgx[i] = new_double(N+1);
  ///////////////////////////////////////////////////
  /* Initialize newed space  */

  for (i = 0; i < N; ++i)
    for (j = 0; j < i; ++j)
      t->C[i][j] = t->B[i][j] = t->B[j][i] = 0.;

  for (i = 0; i < N; ++i) {
    t->B[i][i] = 1.;
    t->C[i][i] = t->rgD[i] = t->sp.rgInitialStds[i] * sqrt(N / trace);
    t->C[i][i] *= t->C[i][i];
    t->rgpc[i] = t->rgps[i] = 0.;
  }

  t->minEW = rgdouMin(t->rgD, N); t->minEW = t->minEW * t->minEW;
  t->maxEW = rgdouMax(t->rgD, N); t->maxEW = t->maxEW * t->maxEW;

  t->maxdiagC=t->C[0][0];
  for(i=1;i<N;++i) if(t->maxdiagC<t->C[i][i]) t->maxdiagC=t->C[i][i];
  t->mindiagC=t->C[0][0];
  for(i=1;i<N;++i) if(t->mindiagC>t->C[i][i]) t->mindiagC=t->C[i][i];

  /* set xmean */
  for (i = 0; i < N; ++i)
    t->rgxmean[i] = t->rgxold[i] = t->sp.xstart[i];
  /* use in case xstart as typicalX */
  if (t->sp.typicalXcase)
    for (i = 0; i < N; ++i)
      t->rgxmean[i] += t->sigma * t->rgD[i] * cmaes_random_Gauss(&t->rand);

  if (strcmp(t->sp.resumefile, "_no_")  != 0) {
    printf("Calling resume_distribution\n");
    cmaes_resume_distribution(t, t->sp.resumefile);
  }

  return (t->publicFitness);
} /* cmaes_init_final() */

double * cmaes_ChangePopSize(cmaes_t * const t,const int newlambda)
{
  int i;
  int N = t->sp.N;
  //deallocate previous population
  for (i = 0; i < t->sp.lambda; ++i)
    free( t->rgrgx[i]);
  free( t->rgrgx);
  free( t->index);
  free( t->publicFitness);
  free( t->rgFuncValue);

  t->sp.lambda = newlambda;
  t->sp.mu = t->sp.lambda/2;
  double s1, s2;

  if(t->sp.weights != NULL)
    free( t->sp.weights);
  t->sp.weights = new_double(t->sp.mu);
  for (i=0; i<t->sp.mu; ++i)
    t->sp.weights[i] = log(t->sp.mu+1.)-log(i+1.);

  /* normalize weights vector and set mueff */
  for (i=0, s1=0, s2=0; i<t->sp.mu; ++i) {
    s1 += t->sp.weights[i];
    s2 += t->sp.weights[i]*t->sp.weights[i];
  }
  t->sp.mueff = s1*s1/s2;
  t->sp.mucov = t->sp.mueff;
  double t1, t2;
  t1 = 2. / ((N+1.4142)*(N+1.4142));
  t2 = (2.*t->sp.mueff-1.) / ((N+2.)*(N+2.)+t->sp.mueff);
  t2 = (t2 > 1) ? 1 : t2;
  t2 = (1./t->sp.mucov) * t1 + (1.-1./t->sp.mucov) * t2;
  t->sp.ccov = t2;

  for (i=0; i<t->sp.mu; ++i)
    t->sp.weights[i] /= s1;

  if(t->sp.mu < 1 || t->sp.mu > t->sp.lambda ||
      (t->sp.mu==t->sp.lambda && t->sp.weights[0]==t->sp.weights[t->sp.mu-1]))
    FATAL("cmaes_readpara_SetWeights(): invalid setting of mu or lambda",0,0,0);

  //allocate new population
  t->publicFitness = new_double(t->sp.lambda);
  t->rgFuncValue = new_double(t->sp.lambda);

  t->index = (int *) new_void(t->sp.lambda, sizeof(int));
  for (i = 0; i < t->sp.lambda; ++i)
    t->index[i] = i; /* should not be necessary */
  t->rgrgx = (double **)new_void(t->sp.lambda, sizeof(double*));
  for (i = 0; i < t->sp.lambda; ++i)
    t->rgrgx[i] = new_double(N+1);

  return t->publicFitness;
}
/* --------------------------------------------------------- */
/* --------------------------------------------------------- */
  double *
cmaes_init(cmaes_t * const t, /* "this" */
    int dimension,
    double * const inxstart,
    double * const inrgstddev, /* initial stds */
    long int inseed,
    int lambda,
    const char *input_parameter_filename)
{
  cmaes_init_para(t, dimension, inxstart, inrgstddev, inseed,
      lambda, input_parameter_filename);
  t->isStuck = 0;
  return cmaes_init_final(t);
}

/* --------------------------------------------------------- */
/* --------------------------------------------------------- */

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

void cmaes_resume_distribution(cmaes_t * const t, char *filename)
{
  int i, j, res, n, d_int;
  double d;
  FILE *fp = fopen( filename, "r");
  if(fp == NULL) {
    ERRORMESSAGE("cmaes_resume_distribution(): could not open '",
        filename, "'",0);
    return;
  }
  /* count number of "resume" entries */
  i = 0; res = 0; int safety = 0;
  while (1) {
    if ((res = fscanf(fp, " resume %d", &d_int)) == EOF)
      break;
    else if (res==0)
      fscanf(fp, " %*s");
    else if(res > 0)
      i += 1;
    if (safety++ > 1e4) abort();
  }

  /* go to last "resume" entry */
  n = i; i = 0; res = 0; rewind(fp);
  while (i<n) {
    if ((res = fscanf(fp, " resume %d", &d_int)) == EOF)
      FATAL("cmaes_resume_distribution(): Unexpected error, bug",0,0,0);
    else if (res==0)
      fscanf(fp, " %*s");
    else if(res > 0)
      ++i;
  }

  if (d_int != t->sp.N)
    FATAL("cmaes_resume_distribution(): Dimension numbers do not match",0,0,0);

  /* find next "xmean" entry */
  printf("Going through a while true\n");
  while (1) {
    if ((res = fscanf(fp, " xmean %lg", &d)) == EOF)
      FATAL("cmaes_resume_distribution(): 'xmean' not found",0,0,0);
    else if (res==0)
      fscanf(fp, " %*s");
    else if(res > 0)
      break;
  }

  /* read xmean */
  t->rgxmean[0] = d; res = 1;
  for(i = 1; i < t->sp.N; ++i)
    res += fscanf(fp, " %lg", &t->rgxmean[i]);
  if (res != t->sp.N)
    FATAL("cmaes_resume_distribution(): xmean: dimensions differ",0,0,0);

  /* find next "path for sigma" entry */
  printf("Going through a while true\n");
  while (1) {
    if ((res = fscanf(fp, " path for sigma %lg", &d)) == EOF)
      FATAL("cmaes_resume_distribution(): 'path for sigma' not found",0,0,0);
    else if (res==0)
      fscanf(fp, " %*s");
    else if(res > 0)
      break;
  }

  /* read ps */
  t->rgps[0] = d; res = 1;
  for(i = 1; i < t->sp.N; ++i)
    res += fscanf(fp, " %lg", &t->rgps[i]);
  if (res != t->sp.N)
    FATAL("cmaes_resume_distribution(): ps: dimensions differ",0,0,0);

  /* find next "path for C" entry */
  printf("Going through a while true\n");
  while (1) {
    if ((res = fscanf(fp, " path for C %lg", &d)) == EOF)
      FATAL("cmaes_resume_distribution(): 'path for C' not found",0,0,0);
    else if (res==0)
      fscanf(fp, " %*s");
    else if(res > 0)
      break;
  }
  /* read pc */
  t->rgpc[0] = d; res = 1;
  for(i = 1; i < t->sp.N; ++i)
    res += fscanf(fp, " %lg", &t->rgpc[i]);
  if (res != t->sp.N)
    FATAL("cmaes_resume_distribution(): pc: dimensions differ",0,0,0);

  /* find next "sigma" entry */
  printf("Going through a while true\n");
  while (1) {
    if ((res = fscanf(fp, " sigma %lg", &d)) == EOF)
      FATAL("cmaes_resume_distribution(): 'sigma' not found",0,0,0);
    else if (res==0)
      fscanf(fp, " %*s");
    else if(res > 0)
      break;
  }
  t->sigma = d;

  /* find next entry "covariance matrix" */
  printf("Going through a while true\n");
  while (1) {
    if ((res = fscanf(fp, " covariance matrix %lg", &d)) == EOF)
      FATAL("cmaes_resume_distribution(): 'covariance matrix' not found",0,0,0);
    else if (res==0)
      fscanf(fp, " %*s");
    else if(res > 0)
      break;
  }
  /* read C */
  t->C[0][0] = d; res = 1;
  for (i = 1; i < t->sp.N; ++i)
    for (j = 0; j <= i; ++j)
      res += fscanf(fp, " %lg", &t->C[i][j]);
  if (res != (t->sp.N*t->sp.N+t->sp.N)/2)
    FATAL("cmaes_resume_distribution(): C: dimensions differ",0,0,0);

  fclose(fp);

  t->flgIniphase = 0;
  t->flgEigensysIsUptodate = 0;
  t->flgresumedone = 1;
  cmaes_UpdateEigensystem(t, 1);

} /* cmaes_resume_distribution() */
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
/* --------------------------------------------------------- */
/* --------------------------------------------------------- */

  void
cmaes_exit(cmaes_t * const t)
{
  int i, N = t->sp.N;
  t->version = NULL;
  /* free(t->signals_filename) */
  t->state = -1; /* not really useful at the moment */
  free( t->rgpc);
  free( t->rgps);
  free( t->rgdTmp);
  free( t->rgBDz);
  free( t->rgxmean);
  free( t->rgxold);
  free( t->rgxbestever);
  free( t->rgout);
  free( t->rgD);
  for (i = 0; i < N; ++i) {
    free( t->C[i]);
    free( t->B[i]);
  }
  for (i = 0; i < t->sp.lambda; ++i)
    free( t->rgrgx[i]);
  free( t->rgrgx);
  free( t->C);
  free( t->B);
  free( t->index);
  free( t->publicFitness);
  free( t->rgFuncValue);
  free( t->arFuncValueHist);
  cmaes_random_exit (&t->rand);
  cmaes_readpara_exit (&t->sp);
} /* cmaes_exit() */


/* --------------------------------------------------------- */
/* --------------------------------------------------------- */
  double const *
cmaes_SetMean(cmaes_t * const t, const double *xmean)
  /*
   * Distribution mean could be changed before SamplePopulation().
   * This might lead to unexpected behaviour if done repeatedly.
   */
{
  int i, N=t->sp.N;

  if (t->state >= 1 && t->state < 3)
    FATAL("cmaes_SetMean: mean cannot be set inbetween the calls of ",
        "SamplePopulation and UpdateDistribution",0,0);

  if (xmean != NULL && xmean != t->rgxmean)
    for(i = 0; i < N; ++i)
      t->rgxmean[i] = xmean[i];
  else
    xmean = t->rgxmean;

  return xmean;
}

/* --------------------------------------------------------- */
/* --------------------------------------------------------- */
// we can modify t->sigma before this function and will not be affected
  double * const *
cmaes_SamplePopulation(cmaes_t * const t)
{
  int iNk, i, j, N=t->sp.N;
  int flgdiag = ((t->sp.diagonalCov == 1) || (t->sp.diagonalCov >= t->gen));
  double sum;
  double const * const xmean = t->rgxmean;

  /* cmaes_SetMean(t, xmean); * xmean could be changed at this point */

  /* calculate eigensystem  */
  if (!t->flgEigensysIsUptodate) {
    if (!flgdiag) {
      cmaes_UpdateEigensystem(t, 0);
    }
    else {
      for (i = 0; i < N; ++i)
        t->rgD[i] = sqrt(t->C[i][i]);
      t->minEW = douSquare(rgdouMin(t->rgD, N));
      t->maxEW = douSquare(rgdouMax(t->rgD, N));
      t->flgEigensysIsUptodate = 1;
      cmaes_timings_start(&t->eigenTimings);
    }
  }
  fflush(0);
  /* treat minimal standard deviations and numeric problems */
  TestMinStdDevs(t);

  for (iNk = 0; iNk < t->sp.lambda; ++iNk)
  { /* generate scaled cmaes_random vector (D * z)    */
    for (i = 0; i < N; ++i) {
      const double sampled = cmaes_random_Gauss(&t->rand);
      //if (sampled != sampled)
      //printf("Sampled %e\n", sampled);

      if (isnan(t->rgD[i])) {perror("SEigenvalue is fucking nan\n"); abort();}

      if (flgdiag)
        t->rgrgx[iNk][i] = xmean[i] + t->sigma * t->rgD[i] * sampled;
      else
        t->rgdTmp[i] = t->rgD[i] * sampled;
    }
    if (!flgdiag)
      /* add mutation (sigma * B * (D*z)) */
      for (i = 0; i < N; ++i) {
        for (j = 0, sum = 0.; j < N; ++j)
          sum += t->B[i][j] * t->rgdTmp[j];
        t->rgrgx[iNk][i] = xmean[i] + t->sigma * sum;
      }
  }
  if(t->state == 3 || t->gen == 0)
    ++t->gen;
  t->state = 1;

  return(t->rgrgx);
} /* SamplePopulation() */

/* --------------------------------------------------------- */
/* --------------------------------------------------------- */
  double const *
cmaes_ReSampleSingle_old( cmaes_t *t, double *rgx)
{
  int i, j, N=t->sp.N;
  double sum;

  if (rgx == NULL)
    FATAL("cmaes_ReSampleSingle(): Missing input double *x",0,0,0);

  for (i = 0; i < N; ++i)
    t->rgdTmp[i] = t->rgD[i] * cmaes_random_Gauss(&t->rand);
  /* add mutation (sigma * B * (D*z)) */
  for (i = 0; i < N; ++i) {
    for (j = 0, sum = 0.; j < N; ++j)
      sum += t->B[i][j] * t->rgdTmp[j];
    rgx[i] = t->rgxmean[i] + t->sigma * sum;
  }
  return rgx;
}

/* --------------------------------------------------------- */
/* --------------------------------------------------------- */
  double * const *
cmaes_ReSampleSingle( cmaes_t * const t, int iindex)
{
  int i, j, N=t->sp.N;
  double *rgx;
  double sum;
  static char s[99];

  if (iindex < 0 || iindex >= t->sp.lambda) {
    sprintf(s, "index==%d must be between 0 and %d", iindex, t->sp.lambda);
    FATAL("cmaes_ReSampleSingle(): Population member ",s,0,0);
  }
  rgx = t->rgrgx[iindex];

  for (i = 0; i < N; ++i)
    t->rgdTmp[i] = t->rgD[i] * cmaes_random_Gauss(&t->rand);
  /* add mutation (sigma * B * (D*z)) */
  for (i = 0; i < N; ++i) {
    for (j = 0, sum = 0.; j < N; ++j)
      sum += t->B[i][j] * t->rgdTmp[j];
    rgx[i] = t->rgxmean[i] + t->sigma * sum;
  }
  return(t->rgrgx);
}

/* --------------------------------------------------------- */
/* --------------------------------------------------------- */
  double *
cmaes_SampleSingleInto( cmaes_t * const t, double * rgx)
{
  int i, j, N=t->sp.N;
  double sum;

  if (rgx == NULL) rgx = new_double(N);

  for (i = 0; i < N; ++i)
    t->rgdTmp[i] = t->rgD[i] * cmaes_random_Gauss(&t->rand);
  /* add mutation (sigma * B * (D*z)) */
  for (i = 0; i < N; ++i) {
    for (j = 0, sum = 0.; j < N; ++j)
      sum += t->B[i][j] * t->rgdTmp[j];
    rgx[i] = t->rgxmean[i] + t->sigma * sum;
  }
  return rgx;
}

/* --------------------------------------------------------- */
/* --------------------------------------------------------- *
   double *
   cmaes_PerturbSolutionInto( cmaes_t *t, double *rgx)
   {
   int i, j, N=t->sp.N;
   double sum;

   if (rgx == NULL)
   rgx = new_double(N);
   if (xmean == NULL)
   FATAL("cmaes_PerturbSolutionInto(): xmean was not given",0,0,0);

   for (i = 0; i < N; ++i)
   t->rgdTmp[i] = t->rgD[i] * cmaes_random_Gauss(&t->rand);
// add mutation (sigma * B * (D*z))
for (i = 0; i < N; ++i) {
for (j = 0, sum = 0.; j < N; ++j)
sum += t->B[i][j] * t->rgdTmp[j];
rgx[i] = xmean[i] + eps * t->sigma * sum;
}
return rgx;
}

 * --------------------------------------------------------- */
/* --------------------------------------------------------- */
  const double *
cmaes_Optimize( cmaes_t * const evo, double(*pFun)(double const *, int dim), long iterations)
  /* TODO: make signals.par another argument or, even better, part of cmaes_t */
{
  const char * signalsFilename = "cmaes_signals.par";
  double *const*pop; /* sampled population */
  const char *stop;
  int i;
  double startiter = evo->gen;

  while(!(stop=cmaes_TestForTermination(evo)) &&
      (evo->gen < startiter + iterations || !iterations))
  {
    /* Generate population of new candidate solutions */
    pop = cmaes_SamplePopulation(evo); /* do not change content of pop */

    /* Compute fitness value for each candidate solution */
    for (i = 0; i < cmaes_Get(evo, "popsize"); ++i) {
      evo->publicFitness[i] = (*pFun)(pop[i], evo->sp.N);
    }

    /* update search distribution */
    cmaes_UpdateDistribution(evo, evo->publicFitness);

    /* read control signals for output and termination */
    if (signalsFilename)
      cmaes_ReadSignals(evo, signalsFilename);
    fflush(stdout);
  } /* while !cmaes_TestForTermination(evo) */

  /* write some data */
  cmaes_WriteToFile(evo, "all", "allcmaes.dat");

  return cmaes_GetPtr(evo, "xbestever");
}


/* --------------------------------------------------------- */
/* --------------------------------------------------------- */
  double *
cmaes_UpdateDistribution( cmaes_t * const t, const double *rgFunVal)
{
  double eps = 1e-16; /* Math.pow(2.0,-52.0);  == 2.22e-16 */
  int i, j, iNk, hsig, N=t->sp.N;
  int flgdiag = ((t->sp.diagonalCov == 1) || (t->sp.diagonalCov >= t->gen));
  double sum;
  double psxps;

  if(t->state == 3)
    FATAL("cmaes_UpdateDistribution(): You need to call \n",
        "SamplePopulation() before update can take place.",0,0);
  if(rgFunVal == NULL)
    FATAL("cmaes_UpdateDistribution(): ",
        "Fitness function value array input is missing.",0,0);

  if(t->state == 1)  /* function values are delivered here */
    t->countevals += t->sp.lambda;
  else
    ERRORMESSAGE("cmaes_UpdateDistribution(): unexpected state",0,0,0);

  /* assign function values */
  for (i=0; i < t->sp.lambda; ++i) {
    if (isnan(rgFunVal[i])) {printf("nan function\n"); abort();}
    t->rgrgx[i][N] = t->rgFuncValue[i] = rgFunVal[i];
  }


  /* Generate index */
  Sorted_index(rgFunVal, t->index, t->sp.lambda);
  if (t->sp.damps<t->sp.cs) {
    printf("sp.damps\n");
    abort();
  }
  /* Test if function values are identical, escape flat fitness */
  if (fabs(t->rgFuncValue[t->index[0]] - t->rgFuncValue[t->index[(int)t->sp.lambda/2]]) < eps) {
    t->sigma *= exp(0.2+t->sp.cs/t->sp.damps);
    printf("Going to have trouble with f %d: sigma %e cs %e damp %e ccov1 %e ccovmu %e ccumcov %e\n",
        t->sp.funcID, t->sigma, t->sp.cs, t->sp.damps, t->sp.ccov1, t->sp.ccovmu, t->sp.ccumcov);
    if (isnan(t->sigma)) {t->sigma = 0; perror("Fucking nan sigma\n");}
    t->isStuck = 1;
    //if(t->sigma<2e-8) t->sigma = 2e-8;
    //if(t->sigma>1e7) t->sigma = 1e7;
    //fflush(0);
    //cmaes_ReadSignals(t, "../cmaes_signals.par");
    ERRORMESSAGE("Warning: sigma increased due to equal function values\n",
        "   Reconsider the formulation of the objective function",0,0);
    //return (t->rgxmean);
  }
  if(t->sigma<eps) t->sigma=eps;
  /* update function value history */
  const int sizeFunHist = 10+(int)ceil(t->sp.N*30.);
  for(i = sizeFunHist-1; i > 0; --i)
    t->arFuncValueHist[i] = t->arFuncValueHist[i-1];
  t->arFuncValueHist[0] = rgFunVal[t->index[0]];

  /* update xbestever */
  if (t->rgxbestever[N] > t->rgrgx[t->index[0]][N] || t->gen == 1)
    for (i = 0; i <= N; ++i) {
      t->rgxbestever[i] = t->rgrgx[t->index[0]][i];
      t->rgxbestever[N+1] = t->countevals;
    }

  /* calculate xmean and rgBDz~N(0,C) */
  for (i = 0; i < N; ++i) {
    t->rgxold[i] = t->rgxmean[i];
    t->rgxmean[i] = 0.;
    for (iNk = 0; iNk < t->sp.mu; ++iNk)
      t->rgxmean[i] += t->sp.weights[iNk] * t->rgrgx[t->index[iNk]][i];
    t->rgBDz[i] = sqrt(t->sp.mueff)*(t->rgxmean[i] - t->rgxold[i])/t->sigma;
  }

  /* calculate z := D^(-1) * B^(-1) * rgBDz into rgdTmp */
  for (i = 0; i < N; ++i) {
    if (!flgdiag)
      for (j = 0, sum = 0.; j < N; ++j)
        sum += t->B[j][i] * t->rgBDz[j];
    else
      sum = t->rgBDz[i];
    const double dmp = t->rgD[i]<0 ? t->rgD[i]-eps : t->rgD[i]+eps;
    t->rgdTmp[i] = sum / dmp;
  }

  /* TODO?: check length of t->rgdTmp and set an upper limit, e.g. 6 stds */
  /* in case of manipulation of arx,
     this can prevent an increase of sigma by several orders of magnitude
     within one step; a five-fold increase in one step can still happen.
     */
  /*
     for (j = 0, sum = 0.; j < N; ++j)
     sum += t->rgdTmp[j] * t->rgdTmp[j];
     if (sqrt(sum) > chiN + 6. * sqrt(0.5)) {
     rgdTmp length should be set to upper bound and hsig should become zero
     }
     */

  /* cumulation for sigma (ps) using B*z */
  for (i = 0; i < N; ++i) {
    if (!flgdiag)
      for (j = 0, sum = 0.; j < N; ++j)
        sum += t->B[i][j] * t->rgdTmp[j];
    else
      sum = t->rgdTmp[i];
    t->rgps[i] = (1. - t->sp.cs) * t->rgps[i] +
      sqrt(t->sp.cs * (2. - t->sp.cs)) * sum;
  }

  /* calculate norm(ps)^2 */
  for (i = 0, psxps = 0.; i < N; ++i)
    psxps += t->rgps[i] * t->rgps[i];

  /* cumulation for covariance matrix (pc) using B*D*z~N(0,C) */
  {
    const double denum = sqrt(1. - pow(1.-t->sp.cs, 2*t->gen)) * t->chiN;
    hsig = sqrt(psxps) / (denum+eps) < 1.4 + 2./(N+1);
  }

  for (i = 0; i < N; ++i) {
    t->rgpc[i] = (1. - t->sp.ccumcov) * t->rgpc[i] +
      hsig * sqrt(t->sp.ccumcov * (2. - t->sp.ccumcov)) * t->rgBDz[i];
  }
  if(t->sp.cs<eps) {
    printf("t->sp.cs<eps\n"); abort();
  }
  if(t->sp.mucov<eps) {
    printf("t->sp.mucov<eps\n"); abort();
  }
  /* stop initial phase */
  if (t->flgIniphase &&
      t->gen > douMin(1/t->sp.cs, 1+N/t->sp.mucov))
  {
    const double denum =  1.-pow((1. - t->sp.cs), t->gen);
    if (psxps / t->sp.damps / (denum+eps) < N * 1.05)
      t->flgIniphase = 0;
  }

  /* update of C  */
  Adapt_C2(t, hsig);


  /* update of sigma */
  t->trace = sqrt(psxps);
  t->sigma *= exp(((sqrt(psxps)/t->chiN)-1.)*t->sp.cs/t->sp.damps);
  if(t->sigma<1e-16) t->sigma = 1e-16;
  if(t->sigma>1e7) t->sigma = 1e7;

  if (isnan(t->sigma)) {
    t->sigma = 0;
    perror("Fucking nan sigma 2\n");
    t->isStuck = 1;
  }
  t->state = 3;

  return (t->rgxmean);

} /* cmaes_UpdateDistribution() */


/* --------------------------------------------------------- */
/* --------------------------------------------------------- */
  static void
Adapt_C2(cmaes_t * const t, int hsig)
{
  double eps = 1e-16;
  int i, j, k, N=t->sp.N;
  int flgdiag = ((t->sp.diagonalCov == 1) || (t->sp.diagonalCov >= t->gen));
  for (i = 0; i < N; ++i)
    for (j = 0; j < N; ++j)
      if(isnan(t->C[i][j]))  t->C[i][j] = 0;

  if (t->sp.ccov != 0. && t->flgIniphase == 0) {

    /* definitions for speeding up inner-most loop */
    double ccov1 = (t->sp.ccov1 > 0) ? t->sp.ccov1 :
      douMin(t->sp.ccov * (1./t->sp.mucov) * (flgdiag ? (N+1.5) / 3. : 1.), 1.);
    double ccovmu = (t->sp.ccovmu > 0) ? t->sp.ccovmu :
      douMin(t->sp.ccov * (1-1./t->sp.mucov)* (flgdiag ? (N+1.5) / 3. : 1.), 1.-ccov1);
    double sigmasquare = t->sigma * t->sigma + eps;
    t->flgEigensysIsUptodate = 0;
    //printf("%f %f %f %f %f\n", ccov1, ccovmu, t->sp.ccumcov, t->sp.cs, sigmasquare);
    /* update covariance matrix */

    for (i = 0; i < N; ++i)
      for (j = flgdiag ? i : 0; j <= i; ++j) {
        const double dampedOld = t->C[i][j]>1e6 ? 1e6 : (t->C[i][j]<-1e6 ? -1e6 : t->C[i][j]);
        t->C[i][j] = (1 - ccov1 - ccovmu) * dampedOld
          + ccov1 * (t->rgpc[i] * t->rgpc[j]
              + (1-hsig)*t->sp.ccumcov*(2.-t->sp.ccumcov) * t->C[i][j]);
        for (k = 0; k < t->sp.mu; ++k) { /* additional rank mu update */
          t->C[i][j] += ccovmu * t->sp.weights[k]
            * (t->rgrgx[t->index[k]][i] - t->rgxold[i])
            * (t->rgrgx[t->index[k]][j] - t->rgxold[j])
            / sigmasquare;
        }
      }
    /* update maximal, minimal. mean diagonal values */
    t->meandiagC = 0;
    t->maxdiagC = t->mindiagC = t->C[0][0];
    for (i = 1; i < N; ++i) {
      if(!isnan(t->C[i][i]))
        t->meandiagC += t->C[i][i];

      if (t->maxdiagC < t->C[i][i])
        t->maxdiagC = t->C[i][i];
      else if (t->mindiagC > t->C[i][i])
        t->mindiagC = t->C[i][i];
    }
    t->meandiagC/=N;
  } /* if ccov... */
}


/* --------------------------------------------------------- */
/* --------------------------------------------------------- */
  static void
TestMinStdDevs(cmaes_t * const t)
  /* increases sigma */
{
  if (t->sp.damps<t->sp.cs) {
    printf("sp.damps 2 %f\n", t->sp.damps);
    abort();
  }
  int i, N = t->sp.N;
  if (t->sp.rgDiffMinChange == NULL)
    return;
  int safety = 0;
  for (i = 0; i < N; ++i)
    while (t->sigma * sqrt(t->C[i][i]) < t->sp.rgDiffMinChange[i] && safety++ < 1e4) {
      t->sigma *= exp(0.05+t->sp.cs/t->sp.damps);
    }
  //
  if(t->sigma<1e-16) t->sigma = 1e-16;
  if(t->sigma>1e7) t->sigma = 1e7;
} /* cmaes_TestMinStdDevs() */


/* --------------------------------------------------------- */
/* --------------------------------------------------------- */
void cmaes_WriteToFile(cmaes_t * const t, const char *key, const char *name)
{
  cmaes_WriteToFileAW(t, key, name, "a"); /* default is append */
}

/* --------------------------------------------------------- */
/* --------------------------------------------------------- */
void cmaes_WriteToFileAW(cmaes_t * const t, const char *key, const char *name,
    const char *appendwrite)
{
  const char *s = "tmpcmaes.dat";
  FILE *fp;

  if (name == NULL)
    name = s;

  fp = fopen( name, appendwrite);

  if(fp == NULL) {
    ERRORMESSAGE("cmaes_WriteToFile(): could not open '", name,
        "' with flag ", appendwrite);
    return;
  }

  if (appendwrite[0] == 'w') {
    /* write a header line, very rudimentary */
    fprintf(fp, "%% # %s (randomSeed=%d, %s)\n", key, t->sp.seed, getTimeStr());
  } else
    if (t->gen > 0 || strncmp(name, "outcmaesfit", 11) != 0)
      cmaes_WriteToFilePtr(t, key, fp); /* do not write fitness for gen==0 */

  fclose(fp);

} /* WriteToFile */

/* --------------------------------------------------------- */
void cmaes_WriteToFilePtr(cmaes_t * const t, const char *key, FILE *fp)

  /* this hack reads key words from input key for data to be written to
   * a file, see file signals.par as input file. The length of the keys
   * is mostly fixed, see key += number in the code! If the key phrase
   * does not match the expectation the output might be strange.  for
   * cmaes_t *t == NULL it solely prints key as a header line. Input key
   * must be zero terminated.
   */
{
  int i, k, N=(t ? t->sp.N : 0);
  char const *keyend; /* *keystart; */
  const char *s = "few";
  if (key == NULL)
    key = s;
  /* keystart = key; for debugging purpose */
  keyend = key + strlen(key);


  printf("Going through a while key < keyend)\n");
  while (key < keyend)
  {
    if (strncmp(key, "axisratio", 9) == 0)
    {
      fprintf(fp, "%.2e", sqrt(t->maxEW/t->minEW));
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "idxminSD", 8) == 0)
    {
      int mini=0; for(i=N-1;i>0;--i) if(t->mindiagC==t->C[i][i]) mini=i;
      fprintf(fp, "%d", mini+1);
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "idxmaxSD", 8) == 0)
    {
      int maxi=0; for(i=N-1;i>0;--i) if(t->maxdiagC==t->C[i][i]) maxi=i;
      fprintf(fp, "%d", maxi+1);
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    /* new coordinate system == all eigenvectors */
    if (strncmp(key, "B", 1) == 0)
    {
      /* int j, index[N]; */
      int j, *iindex=(int*)(new_void(N,sizeof(int))); /* MT */
      Sorted_index(t->rgD, iindex, N); /* should not be necessary, see end of QLalgo2 */
      /* One eigenvector per row, sorted: largest eigenvalue first */
      for (i = 0; i < N; ++i)
        for (j = 0; j < N; ++j)
          fprintf(fp, "%g%c", t->B[j][iindex[N-1-i]], (j==N-1)?'\n':'\t');
      ++key;
      free(iindex); /* MT */
    }
    /* covariance matrix */
    if (strncmp(key, "C", 1) == 0)
    {
      int j;
      for (i = 0; i < N; ++i)
        for (j = 0; j <= i; ++j)
          fprintf(fp, "%g%c", t->C[i][j], (j==i)?'\n':'\t');
      ++key;
    }
    /* (processor) time (used) since begin of execution */
    if (strncmp(key, "clock", 4) == 0)
    {
      cmaes_timings_update(&t->eigenTimings);
      fprintf(fp, "%.1f %.1f",  t->eigenTimings.totaltotaltime,
          t->eigenTimings.tictoctime);
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    /* ratio between largest and smallest standard deviation */
    if (strncmp(key, "stddevratio", 11) == 0) /* std dev in coordinate axes */
    {
      fprintf(fp, "%g", sqrt(t->maxdiagC/t->mindiagC));
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    /* standard deviations in coordinate directions (sigma*sqrt(C[i,i])) */
    if (strncmp(key, "coorstddev", 10) == 0
        || strncmp(key, "stddev", 6) == 0) /* std dev in coordinate axes */
    {
      for (i = 0; i < N; ++i)
        fprintf(fp, "%s%g", (i==0) ? "":"\t", t->sigma*sqrt(t->C[i][i]));
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    /* diagonal of D == roots of eigenvalues, sorted */
    if (strncmp(key, "diag(D)", 7) == 0)
    {
      for (i = 0; i < N; ++i)
        t->rgdTmp[i] = t->rgD[i];
      qsort(t->rgdTmp, (unsigned) N, sizeof(double), &SignOfDiff); /* superfluous */
      for (i = 0; i < N; ++i)
        fprintf(fp, "%s%g", (i==0) ? "":"\t", t->rgdTmp[i]);
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "dim", 3) == 0)
    {
      fprintf(fp, "%d", N);
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "eval", 4) == 0)
    {
      fprintf(fp, "%.0f", t->countevals);
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "few(diag(D))", 12) == 0)/* between four and six axes */
    {
      int add = (int)(0.5 + (N + 1.) / 5.);
      for (i = 0; i < N; ++i)
        t->rgdTmp[i] = t->rgD[i];
      qsort(t->rgdTmp, (unsigned) N, sizeof(double), &SignOfDiff);
      for (i = 0; i < N-1; i+=add)        /* print always largest */
        fprintf(fp, "%s%g", (i==0) ? "":"\t", t->rgdTmp[N-1-i]);
      fprintf(fp, "\t%g\n", t->rgdTmp[0]);        /* and smallest */
      break; /* number of printed values is not determined */
    }
    if (strncmp(key, "fewinfo", 7) == 0) {
      fprintf(fp," Iter Fevals  Function Value         Sigma   ");
      fprintf(fp, "MaxCoorDev MinCoorDev AxisRatio   MinDii   Time in eig\n");
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
    }
    if (strncmp(key, "few", 3) == 0) {
      fprintf(fp, " %ld ", t->gen);
      fprintf(fp, " %5.0f ", t->countevals);
      fprintf(fp, "%.15e", t->rgFuncValue[t->index[0]]);
      fprintf(fp, "  %.2e  %.2e %.2e", t->sigma, t->sigma*sqrt(t->maxdiagC),
          t->sigma*sqrt(t->mindiagC));
      fprintf(fp, "  %.2e  %.2e", sqrt(t->maxEW/t->minEW), sqrt(t->minEW));
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "funval", 6) == 0 || strncmp(key, "fitness", 6) == 0)
    {
      fprintf(fp, "%.15e", t->rgFuncValue[t->index[0]]);
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "fbestever", 9) == 0)
    {
      fprintf(fp, "%.15e", t->rgxbestever[N]); /* f-value */
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "fmedian", 7) == 0)
    {
      fprintf(fp, "%.15e", t->rgFuncValue[t->index[(int)(t->sp.lambda/2)]]);
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "fworst", 6) == 0)
    {
      fprintf(fp, "%.15e", t->rgFuncValue[t->index[t->sp.lambda-1]]);
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "arfunval", 8) == 0 || strncmp(key, "arfitness", 8) == 0)
    {
      for (i = 0; i < N; ++i)
        fprintf(fp, "%s%.10e", (i==0) ? "" : "\t",
            t->rgFuncValue[t->index[i]]);
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "gen", 3) == 0)
    {
      fprintf(fp, "%ld", t->gen);
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "iter", 4) == 0)
    {
      fprintf(fp, "%ld", t->gen);
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "sigma", 5) == 0)
    {
      fprintf(fp, "%.4e", t->sigma);
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "minSD", 5) == 0) /* minimal standard deviation */
    {
      fprintf(fp, "%.4e", sqrt(t->mindiagC));
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "maxSD", 5) == 0)
    {
      fprintf(fp, "%.4e", sqrt(t->maxdiagC));
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "mindii", 6) == 0)
    {
      fprintf(fp, "%.4e", sqrt(t->minEW));
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "0", 1) == 0)
    {
      fprintf(fp, "0");
      ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "lambda", 6) == 0 || strncmp(key, "popsi", 5) == 0 || strncmp(key, "populationsi", 12) == 0)
    {
      fprintf(fp, "%d", t->sp.lambda);
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "N", 1) == 0)
    {
      fprintf(fp, "%d", N);
      ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "resume", 6) == 0)
    {
      fprintf(fp, "\n# resume %d\n", N);
      fprintf(fp, "xmean\n");
      cmaes_WriteToFilePtr(t, "xmean", fp);
      fprintf(fp, "path for sigma\n");
      for(i=0; i<N; ++i)
        fprintf(fp, "%g%s", t->rgps[i], (i==N-1) ? "\n":"\t");
      fprintf(fp, "path for C\n");
      for(i=0; i<N; ++i)
        fprintf(fp, "%g%s", t->rgpc[i], (i==N-1) ? "\n":"\t");
      fprintf(fp, "sigma %g\n", t->sigma);
      /* note than B and D might not be up-to-date */
      fprintf(fp, "covariance matrix\n");
      cmaes_WriteToFilePtr(t, "C", fp);
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
    }
    if (strncmp(key, "xbest", 5) == 0) { /* best x in recent generation */
      for(i=0; i<N; ++i)
        fprintf(fp, "%s%g", (i==0) ? "":"\t", t->rgrgx[t->index[0]][i]);
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "xmean", 5) == 0) {
      for(i=0; i<N; ++i)
        fprintf(fp, "%s%g", (i==0) ? "":"\t", t->rgxmean[i]);
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
      fprintf(fp, "%c", (*key=='+') ? '\t':'\n');
    }
    if (strncmp(key, "all", 3) == 0)
    {
      time_t ti = time(NULL);
      fprintf(fp, "\n# --------- %s\n", asctime(localtime(&ti)));
      fprintf(fp, " N %d\n", N);
      fprintf(fp, " seed %d\n", t->sp.seed);
      fprintf(fp, "function evaluations %.0f\n", t->countevals);
      fprintf(fp, "elapsed (CPU) time [s] %.2f\n", t->eigenTimings.totaltotaltime);
      fprintf(fp, "function value f(x)=%g\n", t->rgrgx[t->index[0]][N]);
      fprintf(fp, "maximal standard deviation %g\n", t->sigma*sqrt(t->maxdiagC));
      fprintf(fp, "minimal standard deviation %g\n", t->sigma*sqrt(t->mindiagC));
      fprintf(fp, "sigma %g\n", t->sigma);
      fprintf(fp, "axisratio %g\n", rgdouMax(t->rgD, N)/rgdouMin(t->rgD, N));
      fprintf(fp, "xbestever found after %.0f evaluations, function value %g\n",
          t->rgxbestever[N+1], t->rgxbestever[N]);
      for(i=0; i<N; ++i)
        fprintf(fp, " %12g%c", t->rgxbestever[i],
            (i%5==4||i==N-1)?'\n':' ');
      fprintf(fp, "xbest (of last generation, function value %g)\n",
          t->rgrgx[t->index[0]][N]);
      for(i=0; i<N; ++i)
        fprintf(fp, " %12g%c", t->rgrgx[t->index[0]][i],
            (i%5==4||i==N-1)?'\n':' ');
      fprintf(fp, "xmean \n");
      for(i=0; i<N; ++i)
        fprintf(fp, " %12g%c", t->rgxmean[i],
            (i%5==4||i==N-1)?'\n':' ');
      fprintf(fp, "Standard deviation of coordinate axes (sigma*sqrt(diag(C)))\n");
      for(i=0; i<N; ++i)
        fprintf(fp, " %12g%c", t->sigma*sqrt(t->C[i][i]),
            (i%5==4||i==N-1)?'\n':' ');
      fprintf(fp, "Main axis lengths of mutation ellipsoid (sigma*diag(D))\n");
      for (i = 0; i < N; ++i)
        t->rgdTmp[i] = t->rgD[i];
      qsort(t->rgdTmp, (unsigned) N, sizeof(double), &SignOfDiff);
      for(i=0; i<N; ++i)
        fprintf(fp, " %12g%c", t->sigma*t->rgdTmp[N-1-i],
            (i%5==4||i==N-1)?'\n':' ');
      fprintf(fp, "Longest axis (b_i where d_ii=max(diag(D))\n");
      k = MaxIdx(t->rgD, N);
      for(i=0; i<N; ++i)
        fprintf(fp, " %12g%c", t->B[i][k], (i%5==4||i==N-1)?'\n':' ');
      fprintf(fp, "Shortest axis (b_i where d_ii=max(diag(D))\n");
      k = MinIdx(t->rgD, N);
      for(i=0; i<N; ++i)
        fprintf(fp, " %12g%c", t->B[i][k], (i%5==4||i==N-1)?'\n':' ');
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
    } /* "all" */

    if (*key == '\0')
      break;
    else if (*key != '+') { /* last key was not recognized */
      ERRORMESSAGE("cmaes_t:WriteToFilePtr(): unrecognized key '", key, "'", 0);
      while (*key != '+' && *key != '\0' && key < keyend)
        ++key;
    }
    while (*key == '+')
      ++key;
  } /* while key < keyend */

  if (key > keyend)
    FATAL("cmaes_t:WriteToFilePtr(): BUG regarding key sequence",0,0,0);

} /* WriteToFilePtr */

/* --------------------------------------------------------- */
  double
cmaes_Get( cmaes_t * const t, char const *s)
{
  int N=t->sp.N;

  if (strncmp(s, "axisratio", 5) == 0) { /* between lengths of longest and shortest principal axis of the distribution ellipsoid */
    return (rgdouMax(t->rgD, N)/rgdouMin(t->rgD, N));
  }
  else if (strncmp(s, "eval", 4) == 0) { /* number of function evaluations */
    return (t->countevals);
  }
  else if (strncmp(s, "fctvalue", 6) == 0
      || strncmp(s, "funcvalue", 6) == 0
      || strncmp(s, "funvalue", 6) == 0
      || strncmp(s, "fitness", 3) == 0) { /* recent best function value */
    return(t->rgFuncValue[t->index[0]]);
  }
  else if (strncmp(s, "fmedian", 6) == 0 ) { /* recent mean function value */
    return(t->rgFuncValue[ t->index[(int)(t->sp.lambda/2)] ]);
  }
  else if (strncmp(s, "fbestever", 7) == 0) { /* ever best function value */
    return(t->rgxbestever[N]);
  }
  else if (strncmp(s, "generation", 3) == 0
      || strncmp(s, "iteration", 4) == 0) {
    return((double)t->gen);
  }
  else if (strncmp(s, "maxeval", 4) == 0
      || strncmp(s, "MaxFunEvals", 8) == 0
      || strncmp(s, "stopMaxFunEvals", 12) == 0) { /* maximal number of function evaluations */
    return(t->sp.stopMaxFunEvals);
  }
  else if (strncmp(s, "maxgen", 4) == 0
      || strncmp(s, "MaxIter", 7) == 0
      || strncmp(s, "stopMaxIter", 11) == 0) { /* maximal number of generations */
    return(ceil(t->sp.stopMaxIter));
  }
  else if (strncmp(s, "maxaxislength", 5) == 0) { /* sigma * max(diag(D)) */
    return(t->sigma * sqrt(t->maxEW));
  }
  else if (strncmp(s, "minaxislength", 5) == 0) { /* sigma * min(diag(D)) */
    return(t->sigma * sqrt(t->minEW));
  }
  else if (strncmp(s, "maxstddev", 4) == 0) { /* sigma * sqrt(max(diag(C))) */
    return(t->sigma * sqrt(t->maxdiagC));
  }
  else if (strncmp(s, "minstddev", 4) == 0) { /* sigma * sqrt(min(diag(C))) */
    return(t->sigma * sqrt(t->mindiagC));
  }
  else if (strncmp(s, "N", 1) == 0 || strcmp(s, "n") == 0 ||
      strncmp(s, "dimension", 3) == 0) {
    return (N);
  }
  else if (strncmp(s, "lambda", 3) == 0
      || strncmp(s, "samplesize", 8) == 0
      || strncmp(s, "popsize", 7) == 0) { /* sample size, offspring population size */
    return(t->sp.lambda);
  }
  else if (strncmp(s, "sigma", 3) == 0) {
    return(t->sigma);
  }
  FATAL( "cmaes_Get(cmaes_t, char const * s): No match found for s='", s, "'",0);
  return(0);
} /* cmaes_Get() */

/* --------------------------------------------------------- */
  double *
cmaes_GetInto( cmaes_t * const t, char const *s, double *res)
{
  int i, N = t->sp.N;
  double const * res0 = cmaes_GetPtr(t, s);
  if (res == NULL)
    res = new_double(N);
  for (i = 0; i < N; ++i)
    res[i] = res0[i];
  return res;
}

/* --------------------------------------------------------- */
  double *
cmaes_GetNew( cmaes_t * const t, char const *s)
{
  return (cmaes_GetInto(t, s, NULL));
}

/* --------------------------------------------------------- */
  const double *
cmaes_GetPtr( cmaes_t * const t, char const *s)
{
  int i, N=t->sp.N;

  /* diagonal of covariance matrix */
  if (strncmp(s, "diag(C)", 7) == 0) {
    for (i = 0; i < N; ++i)
      t->rgout[i] = t->C[i][i];
    return(t->rgout);
  }
  /* diagonal of axis lengths matrix */
  else if (strncmp(s, "diag(D)", 7) == 0) {
    return(t->rgD);
  }
  /* vector of standard deviations sigma*sqrt(diag(C)) */
  else if (strncmp(s, "stddev", 3) == 0) {
    for (i = 0; i < N; ++i)
      t->rgout[i] = t->sigma * sqrt(t->C[i][i]);
    return(t->rgout);
  }
  /* bestever solution seen so far */
  else if (strncmp(s, "xbestever", 7) == 0)
    return(t->rgxbestever);
  /* recent best solution of the recent population */
  else if (strncmp(s, "xbest", 5) == 0)
    return(t->rgrgx[t->index[0]]);
  /* mean of the recent distribution */
  else if (strncmp(s, "xmean", 1) == 0)
    return(t->rgxmean);

  return(NULL);
}

/* --------------------------------------------------------- */
/* tests stopping criteria
 *   returns a string of satisfied stopping criterion for each line
 *   otherwise NULL
 */
  const char *
cmaes_TestForTermination( cmaes_t * const t)
{
  double range, fac;
  int iAchse, iKoo;
  int flgdiag = ((t->sp.diagonalCov == 1) || (t->sp.diagonalCov >= t->gen));
  static char sTestOutString[3024];
  char * cp = sTestOutString;
  int i, cTemp, N=t->sp.N;
  cp[0] = '\0';

  /* function value reached */
  if ((t->gen > 1 || t->state > 1) && t->sp.stStopFitness.flg &&
      t->rgFuncValue[t->index[0]] <= t->sp.stStopFitness.val)
    cp += sprintf(cp, "Fitness: function value %7.2e <= stopFitness (%7.2e)\n",
        t->rgFuncValue[t->index[0]], t->sp.stStopFitness.val);

  /* TolFun */
  const int sizeFunHist = 10+(int)ceil(t->sp.N*30./t->sp.lambda);
  const long histLen = t->gen > sizeFunHist ? sizeFunHist : t->gen; //min
  range = douMax(rgdouMax(t->arFuncValueHist, histLen),
      rgdouMax(t->rgFuncValue, t->sp.lambda)) -
    douMin(rgdouMin(t->arFuncValueHist, histLen),
        rgdouMin(t->rgFuncValue, t->sp.lambda));

  if (t->gen > 0 && range <= t->sp.stopTolFun) {
    cp += sprintf(cp,
        "TolFun: function value differences %7.2e < stopTolFun=%7.2e\n",
        range, t->sp.stopTolFun);
  }

  /* TolFunHist */
  if (t->gen > sizeFunHist) {
    range = rgdouMax(t->arFuncValueHist, sizeFunHist)
      - rgdouMin(t->arFuncValueHist, sizeFunHist);
    if (range <= t->sp.stopTolFunHist)
      cp += sprintf(cp,
          "TolFunHist: history of function value changes %7.2e stopTolFunHist=%7.2e",
          range, t->sp.stopTolFunHist);
  }

  /* TolX */
  for(i=0, cTemp=0; i<N; ++i) {
    cTemp += (t->sigma * sqrt(t->C[i][i]) < t->sp.stopTolX) ? 1 : 0;
    cTemp += (t->sigma * t->rgpc[i] < t->sp.stopTolX) ? 1 : 0;
  }
  if (cTemp == 2*N) {
    cp += sprintf(cp,
        "TolX: object variable changes below %7.2e \n",
        t->sp.stopTolX);
  }

  /* TolUpX */
  for(i=0; i<N; ++i) {
    if (t->sigma * sqrt(t->C[i][i]) > t->sp.stopTolUpXFactor * t->sp.rgInitialStds[i])
      break;
  }
  if (i < N) {
    cp += sprintf(cp,
        "TolUpX: standard deviation increased by more than %7.2e, larger initial standard deviation recommended \n",
        t->sp.stopTolUpXFactor);
  }

  /* Condition of C greater than dMaxSignifKond */
  if (t->maxEW >= t->minEW * t->dMaxSignifKond) {
    cp += sprintf(cp,
        "ConditionNumber: maximal condition number %7.2e reached. maxEW=%7.2e,minEW=%7.2e,maxdiagC=%7.2e,mindiagC=%7.2e\n",
        t->dMaxSignifKond, t->maxEW, t->minEW, t->maxdiagC, t->mindiagC);
  } /* if */

  /* Principal axis i has no effect on xmean, ie.
     x == x + 0.1 * sigma * rgD[i] * B[i] */
  if (!flgdiag) {
    for (iAchse = 0; iAchse < N; ++iAchse)
    {
      fac = 0.1 * t->sigma * t->rgD[iAchse];
      for (iKoo = 0; iKoo < N; ++iKoo){
        if (t->rgxmean[iKoo] != t->rgxmean[iKoo] + fac * t->B[iKoo][iAchse])
          break;
      }
      if (iKoo == N)
      {
        /* t->sigma *= exp(0.2+t->sp.cs/t->sp.damps); */
        cp += sprintf(cp,
            "NoEffectAxis: standard deviation 0.1*%7.2e in principal axis %d without effect\n",
            fac/0.1, iAchse);
        break;
      } /* if (iKoo == N) */
    } /* for iAchse             */
  } /* if flgdiag */
  /* Component of xmean is not changed anymore */
  for (iKoo = 0; iKoo < N; ++iKoo)
  {
    if (t->rgxmean[iKoo] == t->rgxmean[iKoo] +
        0.2*t->sigma*sqrt(t->C[iKoo][iKoo]))
    {
      /* t->C[iKoo][iKoo] *= (1 + t->sp.ccov); */
      /* flg = 1; */
      cp += sprintf(cp,
          "NoEffectCoordinate: standard deviation 0.2*%7.2e in coordinate %d without effect\n",
          t->sigma*sqrt(t->C[iKoo][iKoo]), iKoo);
      break;
    }

  } /* for iKoo */
  /* if (flg) t->sigma *= exp(0.05+t->sp.cs/t->sp.damps); */

  if(t->countevals >= t->sp.stopMaxFunEvals)
    cp += sprintf(cp, "MaxFunEvals: conducted function evaluations %.0f >= %g\n",
        t->countevals, t->sp.stopMaxFunEvals);
  if(t->gen >= t->sp.stopMaxIter)
    cp += sprintf(cp, "MaxIter: number of iterations %ld >= %g\n",
        t->gen, t->sp.stopMaxIter);
  if(t->flgStop)
    cp += sprintf(cp, "Manual: stop signal read\n");

  if (cp - sTestOutString>320)
    ERRORMESSAGE("Bug in cmaes_t:Test(): sTestOutString too short",0,0,0);

  if (cp != sTestOutString) {
    return sTestOutString;
  }

  return(NULL);

} /* cmaes_Test() */

/* --------------------------------------------------------- */
void cmaes_ReadSignals(cmaes_t * const t, char const *filename)
{
  const char *s = "cmaes_signals.par";
  FILE *fp;
  if (filename == NULL)
    filename = s;
  /* if (filename) assign_string(&(t->signalsFilename), filename)*/
  fp = fopen( filename, "r");
  if(fp == NULL) {
    return;
  }
  cmaes_ReadFromFilePtr( t, fp);
  fclose(fp);
}
/* --------------------------------------------------------- */
void cmaes_ReadFromFilePtr( cmaes_t * const t, FILE *fp)
  /* reading commands e.g. from signals.par file
  */
{
  const char *keys[15]; /* key strings for scanf */
  char s[199], sin1[99], sin2[129], sin3[99], sin4[99];
  int ikey, ckeys, nb;
  double d;
  static int flglockprint = 0;
  static int flglockwrite = 0;
  static long countiterlastwritten;
  static long maxdiffitertowrite; /* to prevent long gaps at the beginning */
  int flgprinted = 0;
  int flgwritten = 0;
  double deltaprinttime = (double)(time(NULL)-t->printtime); /* using clock instead might not be a good */
  double deltawritetime = (double)(time(NULL)-t->writetime); /* idea as disc time is not CPU time? */
  double deltaprinttimefirst = (double)(t->firstprinttime ? time(NULL)-t->firstprinttime : 0); /* time is in seconds!? */
  double deltawritetimefirst = (double)(t->firstwritetime ? time(NULL)-t->firstwritetime : 0);
  if (countiterlastwritten > t->gen) { /* probably restarted */
    maxdiffitertowrite = 0;
    countiterlastwritten = 0;
  }

  keys[0] = " stop%98s %98s";        /* s=="now" or eg "MaxIter+" %lg"-number */
  /* works with and without space */
  keys[1] = " print %98s %98s";       /* s==keyword for WriteFile */
  keys[2] = " write %98s %128s %98s"; /* s1==keyword, s2==filename */
  keys[3] = " check%98s %98s";
  keys[4] = " maxTimeFractionForEigendecompostion %98s";
  ckeys = 5;
  strcpy(sin2, "tmpcmaes.dat");

  if (cmaes_TestForTermination(t))
  {
    deltaprinttime = (double)time(NULL); /* forces printing */
    deltawritetime = (double)time(NULL);
  }
  while(fgets(s, sizeof(s), fp) != NULL)
  {
    if (s[0] == '#' || s[0] == '%') /* skip comments  */
      continue;
    sin1[0] = sin2[0] = sin3[0] = sin4[0] = '\0';
    for (ikey=0; ikey < ckeys; ++ikey)
    {
      if((nb=sscanf(s, keys[ikey], sin1, sin2, sin3, sin4)) >= 1)
      {
        switch(ikey) {
          case 0 : /* "stop", reads "stop now" or eg. stopMaxIter */
            if (strncmp(sin1, "now", 3) == 0)
              t->flgStop = 1;
            else if (strncmp(sin1, "MaxFunEvals", 11) == 0) {
              if (sscanf(sin2, " %lg", &d) == 1)
                t->sp.stopMaxFunEvals = d;
            }
            else if (strncmp(sin1, "MaxIter", 4) == 0) {
              if (sscanf(sin2, " %lg", &d) == 1)
                t->sp.stopMaxIter = d;
            }
            else if (strncmp(sin1, "Fitness", 7) == 0) {
              if (sscanf(sin2, " %lg", &d) == 1)
              {
                t->sp.stStopFitness.flg = 1;
                t->sp.stStopFitness.val = d;
              }
            }
            else if (strncmp(sin1, "TolFunHist", 10) == 0) {
              if (sscanf(sin2, " %lg", &d) == 1)
                t->sp.stopTolFunHist = d;
            }
            else if (strncmp(sin1, "TolFun", 6) == 0) {
              if (sscanf(sin2, " %lg", &d) == 1)
                t->sp.stopTolFun = d;
            }
            else if (strncmp(sin1, "TolX", 4) == 0) {
              if (sscanf(sin2, " %lg", &d) == 1)
                t->sp.stopTolX = d;
            }
            else if (strncmp(sin1, "TolUpXFactor", 4) == 0) {
              if (sscanf(sin2, " %lg", &d) == 1)
                t->sp.stopTolUpXFactor = d;
            }
            break;
          case 1 : /* "print" */
            d = 1; /* default */
            if (sscanf(sin2, "%lg", &d) < 1 && deltaprinttimefirst < 1)
              d = 0; /* default at first time */
            if (deltaprinttime >= d && !flglockprint) {
              cmaes_WriteToFilePtr(t, sin1, stdout);
              flgprinted = 1;
            }
            if(d < 0)
              flglockprint += 2;
            break;
          case 2 : /* "write" */
            /* write header, before first generation */
            if (t->countevals < t->sp.lambda && t->flgresumedone == 0)
              cmaes_WriteToFileAW(t, sin1, sin2, "w"); /* overwrite */
            d = 0.9; /* default is one with smooth increment of gaps */
            if (sscanf(sin3, "%lg", &d) < 1 && deltawritetimefirst < 2)
              d = 0; /* default is zero for the first second */
            if(d < 0)
              flglockwrite += 2;
            if (!flglockwrite) {
              if (deltawritetime >= d) {
                cmaes_WriteToFile(t, sin1, sin2);
                flgwritten = 1;
              } else if (d < 1
                  && t->gen-countiterlastwritten > maxdiffitertowrite) {
                cmaes_WriteToFile(t, sin1, sin2);
                flgwritten = 1;
              }
            }
            break;
          case 3 : /* check, checkeigen 1 or check eigen 1 */
            if (strncmp(sin1, "eigen", 5) == 0) {
              if (sscanf(sin2, " %lg", &d) == 1) {
                if (d > 0)
                  t->flgCheckEigen = 1;
                else
                  t->flgCheckEigen = 0;
              }
              else
                t->flgCheckEigen = 0;
            }
            break;
          case 4 : /* maxTimeFractionForEigendecompostion */
            if (sscanf(sin1, " %lg", &d) == 1)
              t->sp.updateCmode.maxtime = d;
            break;
          default :
            break;
        }
        break; /* for ikey */
      } /* if line contains keyword */
    } /* for each keyword */
  } /* while not EOF of signals.par */
  if (t->writetime == 0)
    t->firstwritetime = time(NULL);
  if (t->printtime == 0)
    t->firstprinttime = time(NULL);

  if (flgprinted)
    t->printtime = time(NULL);
  if (flgwritten) {
    t->writetime = time(NULL);
    if (t->gen-countiterlastwritten > maxdiffitertowrite)
      ++maxdiffitertowrite; /* smooth prolongation of writing gaps/intervals */
    countiterlastwritten = (long int) t->gen;
  }
  --flglockprint;
  --flglockwrite;
  flglockprint = (flglockprint > 0) ? 1 : 0;
  flglockwrite = (flglockwrite > 0) ? 1 : 0;
} /*  cmaes_ReadFromFilePtr */

/* ========================================================= */
  static int
Check_Eigen( int N,  double ** const C, double * const diag, double ** const Q)
  /*
     exhaustive test of the output of the eigendecomposition
     needs O(n^3) operations

     writes to error file
     returns number of detected inaccuracies
     */
{
  /* compute Q diag Q^T and Q Q^T to check */
  double eps = 1e-16;
  int i, j, k, res = 0;
  double cc, dd;
  static char s[324];

  for (i=0; i < N; ++i)
    for (j=0; j < N; ++j) {
      for (cc=0.,dd=0., k=0; k < N; ++k) {
        cc += diag[k] * Q[i][k] * Q[j][k];
        dd += Q[i][k] * Q[j][k];
      }
      /* check here, is the normalization the right one? */
      if (fabs(cc - C[i>j?i:j][i>j?j:i])/(sqrt(C[i][i]*C[j][j])+eps) > 1e-10
          && fabs(cc - C[i>j?i:j][i>j?j:i]) > 3e-14) {
        sprintf(s, "%d %d: %.17e %.17e, %e",
            i, j, cc, C[i>j?i:j][i>j?j:i], cc-C[i>j?i:j][i>j?j:i]);
        ERRORMESSAGE("cmaes_t:Eigen(): imprecise result detected ",
            s, 0, 0);
        ++res;
      }
      if (fabs(dd - (i==j)) > 1e-10) {
        sprintf(s, "%d %d %.17e ", i, j, dd);
        ERRORMESSAGE("cmaes_t:Eigen(): imprecise result detected (Q not orthog.)",
            s, 0, 0);
        ++res;
      }
    }
  return res;
}

/* --------------------------------------------------------- */
/* --------------------------------------------------------- */
  void
cmaes_UpdateEigensystem(cmaes_t * const t, int flgforce)
{
  int i, N = t->sp.N;

  cmaes_timings_update(&t->eigenTimings);

  if(flgforce == 0) {
    if (t->flgEigensysIsUptodate == 1)
      return;

    /* return on modulo generation number */
    if (t->sp.updateCmode.flgalways == 0 /* not implemented, always ==0 */
        && t->gen < t->genOfEigensysUpdate + t->sp.updateCmode.modulo
       )
      return;

    /* return on time percentage */
    if (t->sp.updateCmode.maxtime < 1.00
        && t->eigenTimings.tictoctime > t->sp.updateCmode.maxtime * t->eigenTimings.totaltime
        && t->eigenTimings.tictoctime > 0.0002)
      return;
  }
  cmaes_timings_tic(&t->eigenTimings);
  /*
     int j;
     for (j=0;j<N;j++)
     for (i=0;i<N;i++)
     printf("%g ",t->C[i][j]);
     printf("\n");
     */
  Eigen( N, t->C, t->rgD, t->B, t->rgdTmp);

  cmaes_timings_toc(&t->eigenTimings);

  /* find largest and smallest eigenvalue, they are supposed to be sorted anyway */
  t->minEW = rgdouMin(t->rgD, N);
  t->maxEW = rgdouMax(t->rgD, N);
  for (i=0;i<N;i++) if(!isnan(t->rgD[i]))
    t->meanEW += t->rgD[i];
  t->meanEW/=N;

  if (t->flgCheckEigen)
    /* needs O(n^3)! writes, in case, error message in error file */
    i = Check_Eigen( N, t->C, t->rgD, t->B);

#if 0
  /* Limit Condition of C to dMaxSignifKond+1 */
  if (t->maxEW > t->minEW * t->dMaxSignifKond) {
    ERRORMESSAGE("Warning: Condition number of covariance matrix at upper limit.",
        " Consider a rescaling or redesign of the objective function. " ,"","");
    printf("\nWarning: Condition number of covariance matrix at upper limit\n");
    tmp = t->maxEW/t->dMaxSignifKond - t->minEW;
    tmp = t->maxEW/t->dMaxSignifKond;
    t->minEW += tmp;
    for (i=0;i<N;++i) {
      t->C[i][i] += tmp;
      t->rgD[i] += tmp;
    }
  } /* if */
  t->dLastMinEWgroesserNull = minEW;
#endif

  for (i = 0; i < N; ++i)
    t->rgD[i] = t->rgD[i]<=0 ? 0 : sqrt(t->rgD[i]);

  t->flgEigensysIsUptodate = 1;
  t->genOfEigensysUpdate = t->gen;

  return;

} /* cmaes_UpdateEigensystem() */


/* ========================================================= */
  static void
Eigen( int N,  double ** const C, double* const diag, double ** const  Q, double * const rgtmp)
  /*
     Calculating eigenvalues and vectors.
     Input:
     N: dimension.
     C: symmetric (1:N)xN-matrix, solely used to copy data to Q
     niter: number of maximal iterations for QL-Algorithm.
     rgtmp: N+1-dimensional vector for temporal use.
     Output:
     diag: N eigenvalues.
     Q: Columns are normalized eigenvectors.
     */
{
  int i, j, k;

  if (rgtmp == NULL) /* was OK in former versions */
    FATAL("cmaes_t:Eigen(): input parameter double *rgtmp must be non-NULL", 0,0,0);

  /* copy C to Q */
  if (C != Q) {
    for (i=0; i < N; ++i)
      for (j = 0; j <= i; ++j)
        if( isnan(C[i][j]) ) {
          printf("Nan C\n");
          Q[i][j] = Q[j][i] = C[i][j] = 0;
        }
        else {
          Q[i][j] = Q[j][i] = C[i][j];
        }
  }

#if 0
  Householder( N, Q, diag, rgtmp);
  QLalgo( N, diag, Q, 30*N, rgtmp+1);
#else
  //printf("1 %p\n",diag); fflush(0);
  Householder2( N, Q, diag, rgtmp);
  //printf("2 %p\n",diag); fflush(0);
  QLalgo2( N, diag, rgtmp, Q);
  //printf("3 %p\n",diag); fflush(0);
#endif

  for (i=0; i < N; ++i)
    if( isnan(diag[i]) ) {
      for (j = 0; j < N; ++j)
        Q[i][j] =0;
      diag[i] = 0;
    }

  for (i=0; i < N; ++i)
    for (k=0; k < N; ++k)
      if( isnan(Q[i][k]) ) {
        diag[i] = 0;
        for (j = 0; j < N; ++j)
          Q[i][j] =0;
      }
}


/* ========================================================= */
static void
QLalgo2 (int n, double* const d, double* const e, double ** const V) {
  /*
     -> n     : Dimension.
     -> d     : Diagonale of tridiagonal matrix.
     -> e[1..n-1] : off-diagonal, output from Householder
     -> V     : matrix output von Householder
     <- d     : eigenvalues
     <- e     : garbage?
     <- V     : basis of eigenvectors, according to d

     Symmetric tridiagonal QL algorithm, iterative
     Computes the eigensystem from a tridiagonal matrix in roughtly 3N^3 operations

     code adapted from Java JAMA package, function tql2.
     */

  int i, k, l, m, safety = 0;
  double f = 0.0;
  double tst1 = 0.0;
  double eps = 1e-16; /* Math.pow(2.0,-52.0);  == 2.22e-16 */

  /* shift input e */
  for (i = 1; i < n; i++) {
    e[i-1] = e[i];
  }
  e[n-1] = 0.0; /* never changed again */

  for (l = 0; l < n; l++) {

    /* Find small subdiagonal element */

    if (tst1 < fabs(d[l]) + fabs(e[l]))
      tst1 = fabs(d[l]) + fabs(e[l]);
    m = l;
    while (m < n) {
      if (fabs(e[m]) <= eps*tst1) { // eps*tst1
        /* if (fabs(e[m]) + fabs(d[m]+d[m+1]) == fabs(d[m]+d[m+1])) { */
        break;
      }
      m++;
      }
      /* If m == l, d[l] is an eigenvalue, */
      /* otherwise, iterate. */
      if (m>=n) printf("m=n, hint?? \n");

      if (m > l && m<n) {  /* TODO: check the case m == n, should be rejected here!? */
        int iter = 0;
        do { /* while (fabs(e[l]) > eps*tst1); */
          double dl1, h;
          double g = d[l];
          double p = (fabs(e[l]) > eps) ? (d[l+1] - g) / (2.0 * e[l]) : 0;
          //double p = (d[l+1] - g) / (2.0 * e[l]) ;
          double r = myhypot(p, 1.);

          iter = iter + 1;  /* Could check iteration count here */

          /* Compute implicit shift */

          if (p < 0) {
            r = -r -eps;
          }
          d[l] = e[l] / (p + r);
          d[l+1] = e[l] * (p + r);
          dl1 = d[l+1];
          h = g - d[l];
          for (i = l+2; i < n; i++) {
            d[i] -= h;
          }
          f = f + h;

          /* Implicit QL transformation. */

          p = d[m];
          {
            double c = 1.0;
            double c2 = c;
            double c3 = c;
            double el1 = e[l+1];
            double s = 0.0;
            double s2 = 0.0;
            for (i = m-1; i >= l; i--) {
              c3 = c2;
              c2 = c;
              s2 = s;
              g = c * e[i];
              h = c * p;
              r = myhypot(p, e[i]);
              e[i+1] = s * r;
              s = e[i] / (r+eps);
              c = p / (r+eps);
              p = c * d[i] - s * g;
              d[i+1] = h + s * (c * g + s * d[i]);

              /* Accumulate transformation. */

              for (k = 0; k < n; k++) {
                h = V[k][i+1];
                V[k][i+1] = s * V[k][i] + c * h;
                V[k][i] = c * V[k][i] - s * h;
              }
            }
            const double denom = dl1<0 ? dl1-eps : dl1+eps;
            p = -s * s2 * c3 * el1 * e[l] / denom;
            e[l] = s * p;
            d[l] = c * p;
          }

          /* Check for convergence. */

        } while (fabs(e[l]) > eps*tst1 && safety++ < 1e8);
      }
      d[l] = d[l] + f;
      e[l] = 0.0;
    }

    /* Sort eigenvalues and corresponding vectors. */
#if 1
    /* TODO: really needed here? So far not, but practical and only O(n^2) */
    {
      int j;
      double p;
      for (i = 0; i < n-1; i++) {
        k = i;
        p = d[i];
        for (j = i+1; j < n; j++) {
          if (d[j] < p) {
            k = j;
            p = d[j];
          }
        }
        if (k != i) {
          d[k] = d[i];
          d[i] = p;
          for (j = 0; j < n; j++) {
            p = V[j][i];
            V[j][i] = V[j][k];
            V[j][k] = p;
          }
        }
      }
    }
#endif
  } /* QLalgo2 */


  /* ========================================================= */
  static void
    Householder2(int n, double ** const V, double * const d, double * const e) {
      /*
         Householder transformation of a symmetric matrix V into tridiagonal form.
         -> n             : dimension
         -> V             : symmetric nxn-matrix
         <- V             : orthogonal transformation matrix:
         tridiag matrix == V * V_in * V^t
         <- d             : diagonal
         <- e[0..n-1]     : off diagonal (elements 1..n-1)

         code slightly adapted from the Java JAMA package, function private tred2()

*/

      int i,j,k;
      double eps = 1e-16; /* Math.pow(2.0,-52.0);  == 2.22e-16 */

      for (j = 0; j < n; j++) {
        d[j] = V[n-1][j];
      }

      /* Householder reduction to tridiagonal form */

      for (i = n-1; i > 0; i--) {

        /* Scale to avoid under/overflow */

        double scale = 0.0;
        double h = 0.0;
        for (k = 0; k < i; k++)
          scale = scale + fabs(d[k]);


        if (scale < eps) {
          e[i] = d[i-1];
          for (j = 0; j < i; j++) {
            d[j] = V[i-1][j];
            V[i][j] = 0.0;
            V[j][i] = 0.0;
          }
        } else {

          /* Generate Householder vector */
          double f, g, hh;

          for (k = 0; k < i; k++) {
            d[k] /= scale;
            h += d[k] * d[k];
          }
          f = d[i-1];
          g = sqrt(h);
          if (f > 0) {
            g = -g;
          }
          e[i] = scale * g;
          h = h - f * g;
          if (h<0) {
            printf("h is neg\n");
            h -= eps;
          } else {
            h += eps;
          }

          d[i-1] = f - g;
          for (j = 0; j < i; j++) {
            e[j] = 0.0;
          }

          /* Apply similarity transformation to remaining columns */

          for (j = 0; j < i; j++) {
            f = d[j];
            V[j][i] = f;
            g = e[j] + V[j][j] * f;
            for (k = j+1; k <= i-1; k++) {
              g += V[k][j] * d[k];
              e[k] += V[k][j] * f;
            }
            e[j] = g;
          }
          f = 0.0;
          for (j = 0; j < i; j++) {
            e[j] /= h;
            f += e[j] * d[j];
          }
          hh = f / (h + h);
          for (j = 0; j < i; j++) {
            e[j] -= hh * d[j];
          }
          for (j = 0; j < i; j++) {
            f = d[j];
            g = e[j];
            for (k = j; k <= i-1; k++) {
              V[k][j] -= (f * e[k] + g * d[k]);
            }
            d[j] = V[i-1][j];
            V[i][j] = 0.0;
          }
        }
        d[i] = h;
      }

      /* Accumulate transformations */

      for (i = 0; i < n-1; i++) {
        double h;
        V[n-1][i] = V[i][i];
        V[i][i] = 1.0;
        h = d[i+1];
        if (h != 0.0) {
          for (k = 0; k <= i; k++) {
            const double denom = h<0 ? h-eps : h+eps;
            d[k] = V[k][i+1] / denom;
          }
          for (j = 0; j <= i; j++) {
            double g = 0.0;
            for (k = 0; k <= i; k++) {
              g += V[k][i+1] * V[k][j];
            }
            for (k = 0; k <= i; k++) {
              V[k][j] -= g * d[k];
            }
          }
        }
        for (k = 0; k <= i; k++) {
          V[k][i+1] = 0.0;
        }
      }
      for (j = 0; j < n; j++) {
        d[j] = V[n-1][j];
        V[n-1][j] = 0.0;
      }
      V[n-1][n-1] = 1.0;
      e[0] = 0.0;

    } /* Housholder() */


#if 0
  /* ========================================================= */
  static void
    WriteMaxErrorInfo(cmaes_t * const t)
    {
      int i,j, N=t->sp.N;
      char *s = (char *)new_void(200+30*(N+2), sizeof(char)); s[0] = '\0';

      sprintf( s+strlen(s),"\nComplete Info\n");
      sprintf( s+strlen(s)," Gen       %20.12g\n", t->gen);
      sprintf( s+strlen(s)," Dimension %d\n", N);
      sprintf( s+strlen(s)," sigma     %e\n", t->sigma);
      sprintf( s+strlen(s)," lastminEW %e\n",
          t->dLastMinEWgroesserNull);
      sprintf( s+strlen(s)," maxKond   %e\n\n", t->dMaxSignifKond);
      sprintf( s+strlen(s),"     x-vector          rgD     Basis...\n");
      ERRORMESSAGE( s,0,0,0);
      s[0] = '\0';
      for (i = 0; i < N; ++i)
      {
        sprintf( s+strlen(s), " %20.12e", t->rgxmean[i]);
        sprintf( s+strlen(s), " %10.4e", t->rgD[i]);
        for (j = 0; j < N; ++j)
          sprintf( s+strlen(s), " %10.2e", t->B[i][j]);
        ERRORMESSAGE( s,0,0,0);
        s[0] = '\0';
      }
      ERRORMESSAGE( "\n",0,0,0);
      free( s);
    } /* WriteMaxErrorInfo() */
#endif

  /* --------------------------------------------------------- */
  /* --------------- Functions: cmaes_timings_t -------------- */
  /* --------------------------------------------------------- */
  /* cmaes_timings_t measures overall time and times between calls
   * of tic and toc. For small time spans (up to 1000 seconds)
   * CPU time via clock() is used. For large time spans the
   * fall-back to elapsed time from time() is used.
   * cmaes_timings_update() must be called often enough to prevent
   * the fallback. */
  /* --------------------------------------------------------- */
  void
    cmaes_timings_init(cmaes_timings_t *t) {
      t->totaltotaltime = 0;
      cmaes_timings_start(t);
    }
  void
    cmaes_timings_start(cmaes_timings_t *t) {
      t->totaltime = 0;
      t->tictoctime = 0;
      t->lasttictoctime = 0;
      t->istic = 0;
      t->lastclock = clock();
      t->lasttime = time(NULL);
      t->lastdiff = 0;
      t->tictoczwischensumme = 0;
      t->isstarted = 1;
    }

  double
    cmaes_timings_update(cmaes_timings_t *t)
    {
      /* returns time between last call of cmaes_timings_*() and now,
       *    should better return totaltime or tictoctime?
       */
      double diffc, difft;
      clock_t lc = t->lastclock; /* measure CPU in 1e-6s */
      time_t lt = t->lasttime;   /* measure time in s */

      if (t->isstarted != 1)
        FATAL("cmaes_timings_started() must be called before using timings... functions",0,0,0);

      t->lastclock = clock(); /* measures at most 2147 seconds, where 1s = 1e6 CLOCKS_PER_SEC */
      t->lasttime = time(NULL);

      diffc = (double)(t->lastclock - lc) / CLOCKS_PER_SEC; /* is presumably in [-21??, 21??] */
      difft = difftime(t->lasttime, lt);                    /* is presumably an integer */

      t->lastdiff = difft; /* on the "save" side */

      /* use diffc clock measurement if appropriate */
      if (diffc > 0 && difft < 1000)
        t->lastdiff = diffc;

      if (t->lastdiff < 0)
        FATAL("BUG in time measurement", 0, 0, 0);

      t->totaltime += t->lastdiff;
      t->totaltotaltime += t->lastdiff;
      if (t->istic) {
        t->tictoczwischensumme += t->lastdiff;
        t->tictoctime += t->lastdiff;
      }

      return t->lastdiff;
    }

  void
    cmaes_timings_tic(cmaes_timings_t *t) {
      if (t->istic) { /* message not necessary ? */
        ERRORMESSAGE("Warning: cmaes_timings_tic called twice without toc",0,0,0);
        return;
      }
      cmaes_timings_update(t);
      t->istic = 1;
    }

  double
    cmaes_timings_toc(cmaes_timings_t *t) {
      if (!t->istic) {
        ERRORMESSAGE("Warning: cmaes_timings_toc called without tic",0,0,0);
        return -1;
      }
      cmaes_timings_update(t);
      t->lasttictoctime = t->tictoczwischensumme;
      t->tictoczwischensumme = 0;
      t->istic = 0;
      return t->lasttictoctime;
    }

  /* --------------------------------------------------------- */
  /* ---------------- Functions: cmaes_random_t -------------- */
  /* --------------------------------------------------------- */
  /* --------------------------------------------------------- */
  /* X_1 exakt :          0.79788456)  */
  /* chi_eins simuliert : 0.798xx   (seed -3) */
  /*                    +-0.001 */
  /* --------------------------------------------------------- */
  /*
     Gauss() liefert normalverteilte Zufallszahlen
     bei vorgegebenem seed.
     */
  /* --------------------------------------------------------- */
  /* --------------------------------------------------------- */

  long
    cmaes_random_init( cmaes_random_t *t, long unsigned inseed)
    {
      clock_t cloc = clock();

      t->flgstored = 0;
      t->rgrand = (long *) new_void(32, sizeof(long));
      if (inseed < 1) {
        while ((long) (cloc - clock()) == 0); /* TODO: remove this for time critical applications? */
        inseed = (long unsigned)labs((long)(100*time(NULL)+clock()));
      }
      return cmaes_random_Start(t, inseed);
    }

  void
    cmaes_random_exit(cmaes_random_t *t)
    {
      free( t->rgrand);
    }

  /* --------------------------------------------------------- */
  long cmaes_random_Start( cmaes_random_t *t, long unsigned inseed)
  {
    long tmp;
    int i;

    t->flgstored = 0;
    t->startseed = inseed; /* purely for bookkeeping */
    while (inseed > 2e9)
      inseed /= 2; /* prevent infinite loop on 32 bit system */
    if (inseed < 1)
      inseed = 1;
    t->aktseed = inseed;
    for (i = 39; i >= 0; --i)
    {
      tmp = t->aktseed/127773;
      t->aktseed = 16807 * (t->aktseed - tmp * 127773)
        - 2836 * tmp;
      if (t->aktseed < 0) t->aktseed += 2147483647;
      if (i < 32)
        t->rgrand[i] = t->aktseed;
    }
    t->aktrand = t->rgrand[0];
    return inseed;
  }

  /* --------------------------------------------------------- */
  double cmaes_random_Gauss(cmaes_random_t *t)
  {
    double x1, x2, rquad, fac;
    double eps = 1e-16;

    if (t->flgstored)
    {
      t->flgstored = 0;
      return t->hold;
    }
    do
    {
      x1 = 2.0 * cmaes_random_Uniform(t) - 1.0;
      x2 = 2.0 * cmaes_random_Uniform(t) - 1.0;
      rquad = x1*x1 + x2*x2;
    } while(rquad >= 1 || rquad <= eps);
    fac = sqrt(-2.0*log(rquad)/rquad);
    t->flgstored = 1;
    t->hold = fac * x1;
    return fac * x2;
  }

  /* --------------------------------------------------------- */
  double cmaes_random_Uniform( cmaes_random_t *t)
  {
    long tmp;

    tmp = t->aktseed/127773;
    t->aktseed = 16807 * (t->aktseed - tmp * 127773)
      - 2836 * tmp;
    if (t->aktseed < 0)
      t->aktseed += 2147483647;
    tmp = t->aktrand / 67108865;
    t->aktrand = t->rgrand[tmp];
    t->rgrand[tmp] = t->aktseed;
    return (double)(t->aktrand)/(2.147483647e9);
  }

  static char *
    szCat(const char *sz1, const char*sz2,
        const char *sz3, const char *sz4);

  /* --------------------------------------------------------- */
  /* -------------- Functions: cmaes_readpara_t -------------- */
  /* --------------------------------------------------------- */
  void
    cmaes_readpara_init (cmaes_readpara_t * const t,
        const int dim,
        const double *  const inxstart,
        const double *  const inrgsigma,
        const int inseed,
        const int lambda,
        const char * filename)
    {
      int i, N;
      /* TODO: make sure cmaes_readpara_init has not been called already */
      t->filename = NULL; /* set after successful Read */
      t->rgsformat = (const char **) new_void(55, sizeof(char *));
      t->rgpadr = (void **) new_void(55, sizeof(void *));
      t->rgskeyar = (const char **) new_void(11, sizeof(char *));
      t->rgp2adr = (double ***) new_void(11, sizeof(double **));
      t->weigkey = (char *)new_void(7, sizeof(char));

      /* All scalars:  */
      i = 0;
      t->rgsformat[i] = " N %d";        t->rgpadr[i++] = (void *) &t->N;
      t->rgsformat[i] = " seed %d";    t->rgpadr[i++] = (void *) &t->seed;
      t->rgsformat[i] = " stopMaxFunEvals %lg"; t->rgpadr[i++] = (void *) &t->stopMaxFunEvals;
      t->rgsformat[i] = " stopMaxIter %lg"; t->rgpadr[i++] = (void *) &t->stopMaxIter;
      t->rgsformat[i] = " stopFitness %lg"; t->rgpadr[i++]=(void *) &t->stStopFitness.val;
      t->rgsformat[i] = " stopTolFun %lg"; t->rgpadr[i++]=(void *) &t->stopTolFun;
      t->rgsformat[i] = " stopTolFunHist %lg"; t->rgpadr[i++]=(void *) &t->stopTolFunHist;
      t->rgsformat[i] = " stopTolX %lg"; t->rgpadr[i++]=(void *) &t->stopTolX;
      t->rgsformat[i] = " stopTolUpXFactor %lg"; t->rgpadr[i++]=(void *) &t->stopTolUpXFactor;
      t->rgsformat[i] = " lambda %d";      t->rgpadr[i++] = (void *) &t->lambda;
      t->rgsformat[i] = " mu %d";          t->rgpadr[i++] = (void *) &t->mu;
      t->rgsformat[i] = " weights %5s";    t->rgpadr[i++] = (void *) t->weigkey;
      t->rgsformat[i] = " fac*cs %lg";t->rgpadr[i++] = (void *) &t->cs;
      t->rgsformat[i] = " fac*damps %lg";   t->rgpadr[i++] = (void *) &t->damps;
      t->rgsformat[i] = " ccumcov %lg";    t->rgpadr[i++] = (void *) &t->ccumcov;
      t->rgsformat[i] = " mucov %lg";     t->rgpadr[i++] = (void *) &t->mucov;
      t->rgsformat[i] = " fac*ccov %lg";  t->rgpadr[i++]=(void *) &t->ccov;
      t->rgsformat[i] = " diagonalCovarianceMatrix %ld"; t->rgpadr[i++]=(void *) &t->diagonalCov;
      t->rgsformat[i] = " updatecov %lg"; t->rgpadr[i++]=(void *) &t->updateCmode.modulo;
      t->rgsformat[i] = " maxTimeFractionForEigendecompostion %lg"; t->rgpadr[i++]=(void *) &t->updateCmode.maxtime;
      t->rgsformat[i] = " resume %59s";    t->rgpadr[i++] = (void *) t->resumefile;
      t->rgsformat[i] = " fac*maxFunEvals %lg";   t->rgpadr[i++] = (void *) &t->facmaxeval;
      t->rgsformat[i] = " fac*updatecov %lg"; t->rgpadr[i++]=(void *) &t->facupdateCmode;
      t->n1para = i;
      t->n1outpara = i-2; /* disregard last parameters in WriteToFile() */

      /* arrays */
      i = 0;
      t->rgskeyar[i]  = " typicalX %d";   t->rgp2adr[i++] = &t->typicalX;
      t->rgskeyar[i]  = " initialX %d";   t->rgp2adr[i++] = &t->xstart;
      t->rgskeyar[i]  = " initialStandardDeviations %d"; t->rgp2adr[i++] = &t->rgInitialStds;
      t->rgskeyar[i]  = " diffMinChange %d"; t->rgp2adr[i++] = &t->rgDiffMinChange;
      t->n2para = i;

      t->N = dim;
      t->seed = (unsigned) inseed;
      t->xstart = NULL;
      t->typicalX = NULL;
      t->typicalXcase = 0;
      t->rgInitialStds = NULL;
      t->rgDiffMinChange = NULL;
      t->stopMaxFunEvals = 1e5;
      t->stopMaxIter = 1e4;
      t->facmaxeval = 1;
      t->stStopFitness.flg = -1;
      t->stopTolFun = 1e-9;
      t->stopTolFunHist = 1e-10;
      t->stopTolX = 1e-9; /* 1e-11*insigma would also be reasonable */
      t->stopTolUpXFactor = 1e3;

      t->ccov1  = -1.; //means that it uses the formula
      t->ccovmu = -1.; //still no effective change from original code

      t->lambda = lambda;
      t->mu = -1;
      t->mucov = -1;
      t->weights = NULL;
      strcpy(t->weigkey, "log");

      t->cs = -1;
      t->ccumcov = -1;
      t->damps = -1;
      t->ccov = -1;

      t->diagonalCov = 0; /* default is 0, but this might change in future, see below */

      t->updateCmode.modulo = -1;
      t->updateCmode.maxtime = -1;
      t->updateCmode.flgalways = 0;
      t->facupdateCmode = 1;
      strcpy(t->resumefile, "_no_");

      /* filename == NULL invokes default in cmaes_readpara_Read... */
      //if (!isNoneStr(filename) && (!filename || strcmp(filename, "writeonly") != 0)){
      //  printf("About to go to readpara thinghy\n");
      //  cmaes_readpara_ReadFromFile(t, filename);
      //  printf("Done with readpara thinghy\n");
      //}

      if (t->N <= 0)
        t->N = dim;

      N = t->N;
      if (N == 0)
        FATAL("cmaes_readpara_t(): problem dimension N undefined.\n",
            "  (no default value available).",0,0);
      if (t->xstart == NULL && inxstart == NULL && t->typicalX == NULL) {
        ERRORMESSAGE("Error: initialX undefined. typicalX = 0.5...0.5 used.","","","");
        printf("\nError: initialX undefined. typicalX = 0.5...0.5 used.\n");
      }
      if (t->rgInitialStds == NULL && inrgsigma == NULL) {
        /* FATAL("initialStandardDeviations undefined","","",""); */
        ERRORMESSAGE("Error: initialStandardDeviations undefined. 0.3...0.3 used.","","","");
        printf("\nError: initialStandardDeviations undefined. 0.3...0.3 used.\n");
      }

      if (t->xstart == NULL) {
        t->xstart = new_double(N);

        /* put inxstart into xstart */
        if (inxstart != NULL) {
          for (i=0; i<N; ++i) {
            t->xstart[i] = inxstart[i];
            //std::cout << "Initial x " << i << " = " << t->xstart[i] << std::endl;
          }
        }
        /* otherwise use typicalX or default */
        else {
          t->typicalXcase = 1;
          for (i=0; i<N; ++i) {
            t->xstart[i] = (t->typicalX == NULL) ? 0.5 : t->typicalX[i];
            //std::cout << "Initial x " << i << " = " << t->xstart[i] << std::endl;
          }
        }
      } /* xstart == NULL */

      if (t->rgInitialStds == NULL) {
        t->rgInitialStds = new_double(N);
        for (i=0; i<N; ++i) {
          t->rgInitialStds[i] = (inrgsigma == NULL) ? 0.3 : inrgsigma[i];
          //std::cout << "Initial std " << i << " = " << t->rgInitialStds[i] << std::endl;
        }
      }

      t->flgsupplemented = 0;


    } /* cmaes_readpara_init */

  /* --------------------------------------------------------- */
  /* --------------------------------------------------------- */
  void cmaes_readpara_exit(cmaes_readpara_t * const t)
  {
    if (t->filename != NULL)
      free( t->filename);
    if (t->xstart != NULL) /* not really necessary */
      free( t->xstart);
    if (t->typicalX != NULL)
      free( t->typicalX);
    if (t->rgInitialStds != NULL)
      free( t->rgInitialStds);
    if (t->rgDiffMinChange != NULL)
      free( t->rgDiffMinChange);
    if (t->weights != NULL)
      free( t->weights);

    free((void*)t->rgsformat);
    free(t->rgpadr);
    free((void*)t->rgskeyar);
    free(t->rgp2adr);
    free(t->weigkey);
  }

  /* --------------------------------------------------------- */
  /* --------------------------------------------------------- */
  void
    cmaes_readpara_ReadFromFile(cmaes_readpara_t * const t, const char * filename)
    {
      char s[1000];
      const char *ss = "cmaes_initials.par";
      int ipara, i;
      int size;
      FILE *fp;
      if (filename == NULL) {
        filename = ss;
      }
      t->filename = NULL; /* nothing read so far */
      fp = fopen( filename, "r");
      if (fp == NULL) {
        ERRORMESSAGE("cmaes_ReadFromFile(): could not open '", filename, "'",0);
        return;
      }
      for (ipara=0; ipara < t->n1para; ++ipara)
      {
        rewind(fp);
        while(fgets(s, sizeof(s), fp) != NULL)
        { /* skip comments  */
          if (s[0] == '#' || s[0] == '%')
            continue;
          if(sscanf(s, t->rgsformat[ipara], t->rgpadr[ipara]) == 1) {
            if (strncmp(t->rgsformat[ipara], " stopFitness ", 13) == 0)
              t->stStopFitness.flg = 1;
            break;
          }
        }
      } /* for */
      if (t->N <= 0)
        FATAL("cmaes_readpara_ReadFromFile(): No valid dimension N",0,0,0);
      for (ipara=0; ipara < t->n2para; ++ipara)
      {
        rewind(fp);
        while(fgets(s, sizeof(s), fp) != NULL) /* read one line */
        { /* skip comments  */
          if (s[0] == '#' || s[0] == '%')
            continue;
          if(sscanf(s, t->rgskeyar[ipara], &size) == 1) { /* size==number of values to be read */
            if (size > 0) {
              *t->rgp2adr[ipara] = new_double(t->N);
              for (i=0;i<size&&i<t->N;++i) /* start reading next line */
                if (fscanf(fp, " %lf", &(*t->rgp2adr[ipara])[i]) != 1)
                  break;
              if (i<size && i < t->N) {
                ERRORMESSAGE("cmaes_readpara_ReadFromFile ", filename, ": ",0);
                FATAL( "'", t->rgskeyar[ipara],
                    "' not enough values found.\n",
                    "   Remove all comments between numbers.");
              }
              for (; i < t->N; ++i) /* recycle */
                (*t->rgp2adr[ipara])[i] = (*t->rgp2adr[ipara])[i%size];
            }
          }
        }
      } /* for */
      fclose(fp);
      assign_string(&(t->filename), filename); /* t->filename must be freed */
      return;
    } /* cmaes_readpara_ReadFromFile() */

  /* --------------------------------------------------------- */
  /* --------------------------------------------------------- */
  void
    cmaes_readpara_WriteToFile(cmaes_readpara_t * const t, const char *filenamedest)
    {
      int ipara, i;
      size_t len;
      time_t ti = time(NULL);
      FILE *fp = fopen( filenamedest, "a");
      if(fp == NULL) {
        ERRORMESSAGE("cmaes_WriteToFile(): could not open '",
            filenamedest, "'",0);
        return;
      }
      fprintf(fp, "\n# Read from %s at %s\n", t->filename ? t->filename : "",
          asctime(localtime(&ti))); /* == ctime() */
      for (ipara=0; ipara < 1; ++ipara) {
        fprintf(fp, t->rgsformat[ipara], *(int *)t->rgpadr[ipara]);
        fprintf(fp, "\n");
      }
      for (ipara=0; ipara < t->n2para; ++ipara) {
        if(*t->rgp2adr[ipara] == NULL)
          continue;
        fprintf(fp, t->rgskeyar[ipara], t->N);
        fprintf(fp, "\n");
        for (i=0; i<t->N; ++i)
          fprintf(fp, "%7.3g%c", (*t->rgp2adr[ipara])[i], (i%5==4)?'\n':' ');
        fprintf(fp, "\n");
      }
      for (ipara=1; ipara < t->n1outpara; ++ipara) {
        if (strncmp(t->rgsformat[ipara], " stopFitness ", 13) == 0)
          if(t->stStopFitness.flg == 0) {
            fprintf(fp, " stopFitness\n");
            continue;
          }
        len = strlen(t->rgsformat[ipara]);
        if (t->rgsformat[ipara][len-1] == 'd') /* read integer */
          fprintf(fp, t->rgsformat[ipara], *(int *)t->rgpadr[ipara]);
        else if (t->rgsformat[ipara][len-1] == 's') /* read string */
          fprintf(fp, t->rgsformat[ipara], (char *)t->rgpadr[ipara]);
        else {
          if (strncmp(" fac*", t->rgsformat[ipara], 5) == 0) {
            fprintf(fp, " ");
            fprintf(fp, t->rgsformat[ipara]+5, *(double *)t->rgpadr[ipara]);
          } else
            fprintf(fp, t->rgsformat[ipara], *(double *)t->rgpadr[ipara]);
        }
        fprintf(fp, "\n");
      } /* for */
      fprintf(fp, "\n");
      fclose(fp);
    } /* cmaes_readpara_WriteToFile() */

  /* --------------------------------------------------------- */
  /* --------------------------------------------------------- */
  void
    cmaes_readpara_SupplementDefaults(cmaes_readpara_t * const t)
    /* Called (only) once to finally set parameters. The settings
     * typically depend on the current parameter values itself,
     * where 0 or -1 may indicate to set them to a certain default
     * value. For this reason calling `SupplementDefaults` twice
     * might lead to unexpected results.
     */
    {
      double t1, t2;
      int N = t->N;
      clock_t cloc = clock();

      if (t->flgsupplemented)
        FATAL("cmaes_readpara_SupplementDefaults() cannot be called twice.",0,0,0);
      if (t->seed < 1) {
        while ((int) (cloc - clock()) == 0)
          ; /* TODO: remove this for time critical applications!? */
        t->seed = (unsigned int)labs((long)(100*time(NULL)+clock()));
      }

      if (t->stStopFitness.flg == -1)
        t->stStopFitness.flg = 0;

      if (t->lambda < 2)
        t->lambda = 4+(int)(3*log((double)N));
      if (t->mu == -1) {
        t->mu = t->lambda/2;
        cmaes_readpara_SetWeights(t, t->weigkey);
      }
      if (t->weights == NULL)
        cmaes_readpara_SetWeights(t, t->weigkey);

      t->cs = (t->mueff + 2.) / (N + t->mueff + 3.);

      t->ccumcov = 4. / (N + 4);

      if (t->mucov < 1) {
        t->mucov = t->mueff;
      }
      t1 = 2. / ((N+1.4142)*(N+1.4142));
      t2 = (2.*t->mueff-1.) / ((N+2.)*(N+2.)+t->mueff);
      t2 = (t2 > 1) ? 1 : t2;
      t2 = (1./t->mucov) * t1 + (1.-1./t->mucov) * t2;
      if (t->ccov >= 0) /* ccov holds the read factor */
        t->ccov *= t2;
      if (t->ccov < 0 || t->ccov > 1) /* set default in case */
        t->ccov = t2;

      if (t->diagonalCov == -1)
        t->diagonalCov = 2 + 100. * N / sqrt((double)t->lambda);

      t->ccov1  = 2. /( pow(N+1.3, 2) + t->mueff );
      t->ccovmu = 2. * (t->mueff-2+1./t->mueff) / ( pow(N+2.,2) + t->mueff );

      if (t->stopMaxFunEvals < 0)  /* may depend on ccov in near future */
        t->stopMaxFunEvals = t->facmaxeval*900*(N+3)*(N+3);
      else
        t->stopMaxFunEvals *= t->facmaxeval;

      if (t->stopMaxIter < 0)
        t->stopMaxIter = ceil((double)(t->stopMaxFunEvals / t->lambda));

      if (t->damps < 0)
        t->damps = 1; /* otherwise a factor was read */
      t->damps = t->damps
        * (1 + 2*douMax(0., sqrt((t->mueff-1.)/(N+1.)) - 1))     /* basic factor */
        * douMax(0.3, 1. -                                       /* modify for short runs */
            (double)N / (1e-6+douMin(t->stopMaxIter, t->stopMaxFunEvals/t->lambda)))
        + t->cs;                                                 /* minor increment */

      if (t->updateCmode.modulo < 0)
        t->updateCmode.modulo = 1./t->ccov/(double)(N)/10.;
      t->updateCmode.modulo *= t->facupdateCmode;
      if (t->updateCmode.maxtime < 0)
        t->updateCmode.maxtime = 0.20; /* maximal 20% of CPU-time */

      t->flgsupplemented = 1;

    } /* cmaes_readpara_SupplementDefaults() */


  /* --------------------------------------------------------- */
  /* --------------------------------------------------------- */
  void
    cmaes_readpara_SetWeights(cmaes_readpara_t * const t, const char * mode)
    {
      double s1, s2;
      int i;

      if(t->weights != NULL)
        free( t->weights);
      t->weights = new_double(t->mu);
      if (strcmp(mode, "lin") == 0)
        for (i=0; i<t->mu; ++i)
          t->weights[i] = t->mu - i;
      else if (strncmp(mode, "equal", 3) == 0)
        for (i=0; i<t->mu; ++i)
          t->weights[i] = 1;
      else if (strcmp(mode, "log") == 0)
        for (i=0; i<t->mu; ++i)
          t->weights[i] = log(t->mu+1.)-log(i+1.);
      else
        for (i=0; i<t->mu; ++i)
          t->weights[i] = log(t->mu+1.)-log(i+1.);

      /* normalize weights vector and set mueff */
      for (i=0, s1=0, s2=0; i<t->mu; ++i) {
        s1 += t->weights[i];
        s2 += t->weights[i]*t->weights[i];
      }
      t->mueff = s1*s1/s2;
      for (i=0; i<t->mu; ++i)
        t->weights[i] /= s1;

      if(t->mu < 1 || t->mu > t->lambda ||
          (t->mu==t->lambda && t->weights[0]==t->weights[t->mu-1]))
        FATAL("cmaes_readpara_SetWeights(): invalid setting of mu or lambda",0,0,0);

    } /* cmaes_readpara_SetWeights() */

  /* --------------------------------------------------------- */
  /* --------------------------------------------------------- */
  static int
    isNoneStr(const char * filename)
    {
      if (filename && (strcmp(filename, "no") == 0
            || strcmp(filename, "non") == 0
            || strcmp(filename, "none") == 0))
        return 1;

      return 0;
    }

  /* --------------------------------------------------------- */
  /* --------------------------------------------------------- */
  static double
    douSquare(double d)
    {
      return d*d;
    }
  static int
    intMin( int i, int j)
    {
      return i < j ? i : j;
    }
  static double
    douMax( double i, double j)
    {
      return i > j ? i : j;
    }
  static double
    douMin( double i, double j)
    {
      return i < j ? i : j;
    }
  static double
    rgdouMax( const double *rgd, int len)
    {
      int i;
      double max = rgd[0];
      for (i = 1; i < len; ++i)
        max = (max < rgd[i]) ? rgd[i] : max;
      return max;
    }

  static double
    rgdouMin( const double *rgd, int len)
    {
      int i;
      double min = rgd[0];
      for (i = 1; i < len; ++i)
        min = (min > rgd[i]) ? rgd[i] : min;
      return min;
    }

  static int
    MaxIdx( const double *rgd, int len)
    {
      int i, res;
      for(i=1, res=0; i<len; ++i)
        if(rgd[i] > rgd[res])
          res = i;
      return res;
    }
  static int
    MinIdx( const double *rgd, int len)
    {
      int i, res;
      for(i=1, res=0; i<len; ++i)
        if(rgd[i] < rgd[res])
          res = i;
      return res;
    }

  static double
    myhypot(double a, double b)
    /* sqrt(a^2 + b^2) numerically stable. */
    {
      double eps = 1e-16;
      double _min = (fabs(a) > fabs(b)) ? fabs(b) : fabs(a);
      double _max = (fabs(a) > fabs(b)) ? fabs(a) : fabs(b);
      double ratio = _min/(_max+eps);
      return _max*sqrt(1+ratio*ratio);
    }

  static int SignOfDiff(const void *d1, const void * d2)
  {
    return *((double *) d1) > *((double *) d2) ? 1 : -1;
  }

#if 1
  /* dirty index sort */
  static void Sorted_index(const double *rgFunVal, int *iindex, int n)
  {
    int i, j;
    for (i=1, iindex[0]=0; i<n; ++i) {
      for (j=i; j>0; --j) {
        if (rgFunVal[iindex[j-1]] < rgFunVal[i])
          break;
        iindex[j] = iindex[j-1]; /* shift up */
      }
      iindex[j] = i; /* insert i */
    }
  }
#endif

  static void * new_void(int n, size_t size)
  {
    static char s[70];
    void *p = calloc((unsigned) n, size);
    if (p == NULL) {
      sprintf(s, "new_void(): calloc(%ld,%ld) failed",(long)n,(long)size);
      FATAL(s,0,0,0);
    }
    return p;
  }

  double *
    cmaes_NewDouble(int n)
    {
      return new_double(n);
    }

  static double * new_double(int n)
  {
    static char s[170];
    double *p = (double *) calloc((unsigned) n, sizeof(double));
    if (p == NULL) {
      sprintf(s, "new_double(): calloc(%ld,%ld) failed",
          (long)n,(long)sizeof(double));
      FATAL(s,0,0,0);
    }
    return p;
  }

  static char * new_string(const char *ins)
  {
    static char s[170];
    unsigned i;
    char *p;
    unsigned len = (unsigned) ((ins != NULL) ? strlen(ins) : 0);

    if (len > 1000) {
      FATAL("new_string(): input string length was larger then 1000 ",
          "(possibly due to uninitialized char *filename)",0,0);
    }

    p = (char *) calloc( len + 1, sizeof(char));
    if (p == NULL) {
      sprintf(s, "new_string(): calloc(%ld,%ld) failed",
          (long)len,(long)sizeof(char));
      FATAL(s,0,0,0);
    }
    for (i = 0; i < len; ++i)
      p[i] = ins[i];
    return p;
  }
  static void assign_string(char ** pdests, const char *ins)
  {
    if (*pdests)
      free(*pdests);
    if (ins == NULL)
      *pdests = NULL;
    else
      *pdests = new_string(ins);
  }

  /* --------------------------------------------------------- */
  /* --------------------------------------------------------- */

  /* ========================================================= */
  void
    cmaes_FATAL(char const *s1, char const *s2, char const *s3,
        char const *s4)
    {
      time_t t = time(NULL);
      ERRORMESSAGE( s1, s2, s3, s4);
      ERRORMESSAGE("*** Exiting cmaes_t ***",0,0,0);
      printf("\n -- %s %s\n", asctime(localtime(&t)),
          s2 ? szCat(s1, s2, s3, s4) : s1);
      printf(" *** CMA-ES ABORTED, see errcmaes.err *** \n");
      fflush(stdout);
      exit(1);
    }

  /* ========================================================= */
  static void
    FATAL(char const *s1, char const *s2, char const *s3,
        char const *s4)
    {
      cmaes_FATAL(s1, s2, s3, s4);
    }

  /* ========================================================= */
  static void ERRORMESSAGE( char const *s1, char const *s2,
      char const *s3, char const *s4)
  {
#if 1
    /*  static char szBuf[700];  desirable but needs additional input argument
      sprintf(szBuf, "%f:%f", gen, gen*lambda);
      */
    time_t t = time(NULL);
    FILE *fp = fopen( "errcmaes.err", "a");
    if (!fp)
    {
      printf("\nFATAL ERROR: %s\n", s2 ? szCat(s1, s2, s3, s4) : s1);
      printf("cmaes_t could not open file 'errcmaes.err'.");
      printf("\n *** CMA-ES ABORTED *** ");
      fflush(stdout);
      exit(1);
    }
    fprintf( fp, "\n -- %s %s\n", asctime(localtime(&t)),
        s2 ? szCat(s1, s2, s3, s4) : s1);
    fclose (fp);
#endif
  }

  /* ========================================================= */
  static char *szCat(const char *sz1, const char*sz2,
      const char *sz3, const char *sz4)
  {
    static char szBuf[700];

    if (!sz1)
      FATAL("szCat() : Invalid Arguments",0,0,0);

    strncpy ((char *)szBuf, sz1, (unsigned)intMin( (int)strlen(sz1), 698));
    szBuf[intMin( (int)strlen(sz1), 698)] = '\0';
    if (sz2)
      strncat ((char *)szBuf, sz2,
          (unsigned)intMin((int)strlen(sz2)+1, 698 - (int)strlen((char const *)szBuf)));
    if (sz3)
      strncat((char *)szBuf, sz3,
          (unsigned)intMin((int)strlen(sz3)+1, 698 - (int)strlen((char const *)szBuf)));
    if (sz4)
      strncat((char *)szBuf, sz4,
          (unsigned)intMin((int)strlen(sz4)+1, 698 - (int)strlen((char const *)szBuf)));
    return (char *) szBuf;
  }
