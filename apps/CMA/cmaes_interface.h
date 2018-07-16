/* --------------------------------------------------------- */
/* --- File: cmaes_interface.h - Author: Nikolaus Hansen --- */
/* ---------------------- last modified:  IV 2007        --- */
/* --------------------------------- by: Nikolaus Hansen --- */
/* --------------------------------------------------------- */
/*
     CMA-ES for non-linear function minimization.

     Copyright (C) 1996, 2003, 2007 Nikolaus Hansen.
     e-mail: hansen AT lri.fr

     Documentation: see file docfunctions.txt

     License: see file cmaes.c
*/
#include "cmaes.h"

/* --------------------------------------------------------- */
/* ------------------ Interface ---------------------------- */
/* --------------------------------------------------------- */

#ifdef __cplusplus
extern "C" {
#endif

/* --- initialization, constructors, destructors --- */
double * cmaes_init(cmaes_t * const, int dimension , double * const xstart, double * const stddev, long seed, int lambda, const char *input_parameter_filename);
void cmaes_init_para(cmaes_t * const, int dimension , double * const xstart, double * const stddev, long seed, int lambda, const char *input_parameter_filename);
double * cmaes_init_final(cmaes_t * const);
void cmaes_resume_distribution(cmaes_t * const evo_ptr, char *filename);
void cmaes_exit(cmaes_t * const);

/* --- core functions --- */
double * const * cmaes_SamplePopulation(cmaes_t * const);
double *         cmaes_UpdateDistribution(cmaes_t * const, const double * const rgFitnessValues);
const char *     cmaes_TestForTermination(cmaes_t * const);

/* --- additional functions --- */
double * const * cmaes_ReSampleSingle( cmaes_t * const t, int index);
double const *   cmaes_ReSampleSingle_old(cmaes_t * const , double *rgx);
double *         cmaes_SampleSingleInto( cmaes_t *  const t, double *rgx);
void             cmaes_UpdateEigensystem(cmaes_t * const, int flgforce);
double *         cmaes_ChangePopSize(cmaes_t * const t,const int newlambda);
/* --- getter functions --- */
double         cmaes_Get(cmaes_t * const, char const *keyword);
const double * cmaes_GetPtr(cmaes_t * const, char const *keyword); /* e.g. "xbestever" */
double *       cmaes_GetNew( cmaes_t * const t, char const *keyword); /* user is responsible to free */
double *       cmaes_GetInto( cmaes_t * const t, char const *keyword, double *mem); /* allocs if mem==NULL, user is responsible to free */

/* --- online control and output --- */
void           cmaes_ReadSignals(cmaes_t * const, char const *filename);
void           cmaes_WriteToFile(cmaes_t * const, const char *szKeyWord, const char *output_filename);
char *         cmaes_SayHello(cmaes_t * const);
/* --- misc --- */
double *       cmaes_NewDouble(int n); /* user is responsible to free */

#ifdef __cplusplus
} // end extern "C"
#endif
