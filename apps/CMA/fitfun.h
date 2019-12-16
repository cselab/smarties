
#ifndef FITFUN_H
#define FITFUN_H

typedef enum {
  ACKLEY,  /* [-32.768, 32.768]; f(x*) = 0, x*[i] = 0 */
  DIXON_PRICE,  /* [-10, 10]; f(x*) = 0, x*[i] = pow(2, -1.+1./pow(2, i)) */
  GRIEWANK,  /* [-600, 600]; f(x*) = 0, x*[i] = 0 */
  LEVY,  /* [-10, 10]; f(x*) = 0, x*[i] = 1 */
  PERM,  /* [-N, N]; f(x*) = 0, x*[i] = i+1. */
  PERM0,  /* [-N, N]; f(x*) = 0, x*[i] = 1./(i+1.) */
  RASTRIGIN,  /* [-5.12, 5.12]; f(x*) = 0, x*[i] = 0 */
  ROSENBROCK,  /* [-5, 10]; f(x*) = 0, x*[i] = 1 */
  ROTATED_HYPER_ELLIPSOID,  /* [-65.536, 65.536]; f(x*) = 0, x*[i] = 0 */
  SCHWEFEL,  /* [-500, 500]; f(x*) = 0, x*[i] = 420.9687 */
  SPHERE,  /* [-5.12, 5.12]; f(x*) = 0, x*[i] = 0 */
  STYBLINSKI_TANG,  /* [-5, 5]; f(x*) = 0, x*[i] = -2.903534 */
  SUM_OF_POWER,  /* [-1, 1]; f(x*) = 0, x*[i] = 0 */
  SUM_OF_SQUARES, /* [-10, 10]; f(x*) = 0, x*[i] = 0 */
  ZAKHAROV,  /* [-5, 10]; f(x*) = 0, x*[i] = 0 */
  _COUNT
} function;



/* the objective (fitness) function to be minimized */
void fitfun(double * const x, int N, double* const output, int * const info);

/* the upper and lower bounds are defined by the function */
void get_upper_lower_bounds(double*const lower_bound, double*const upper_bound,
                  int N, int * const info);

/* final evaluation of the found optimum */
double eval_distance_from_optimum(const double* const found_optimum,
                    int N, int* const info);
#endif
