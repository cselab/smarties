#include <math.h>
#include <cmath>

#include "gsl_headers.h"
#include "fitfun.h"


/* info[0] chooses a random function */
void fitfun(double * const x, int N, double*const output, int * const info)  {
  double f;

  int rnd = info[0] % _COUNT;  /* this defines which function to use */
  //std::cout << "Function evaluation at ";
  //for (int i = 0; i < N; ++i) std::cout << x[i] << " ";
  //std::cout << std::endl;

  switch (rnd) {
    case ACKLEY: { // 0
             double a = 20, b = .2, c = 2.*M_PI, s1 = 0., s2 = 0.;
             for (int i = 0; i < N; ++i) {
               s1 += x[i]*x[i];
               s2 += cos(c*x[i]);
             }
             f = -a*exp(-b*sqrt(s1/N)) - exp(s2/N) + a + exp(1.);
             break;
           }

    case DIXON_PRICE: { // 1
                double s = 0.;
                for (int i = 1; i < N; ++i)
                  s += (i+1.)*pow(2*x[i]*x[i]-x[i-1], 2);
                f = pow(x[0]-1., 2) + s;
                break;
              }

    case GRIEWANK: { // 2
               double s = 0., p = 1.;
               for (int i = 0; i < N; ++i) {
                 s += x[i]*x[i];
                 p *= cos(x[i]/sqrt(1.+i));
               }
               f = s/4000. - p + 1.;
               break;
             }

    case LEVY: { // 3
             double s = 0.;
             for (int i = 0; i < N-1; ++i) {
               s += .0625*pow(x[i]-1, 2)
                 * (1.+10.*pow(sin(M_PI*.25*(3+x[i]) +1), 2));
             }
             f = pow(sin(M_PI*.25*(3.+x[0])), 2) + s
               + .0625*pow(x[N-1]-1, 2)*(1+pow(sin(M_PI*.5*(3+x[N-1])), 2));
             break;
           }

    case PERM: { // 4
             double beta = .5;
             double s2 = 0.;
             for (int i = 0; i < N; ++i) {
               double s1 = 0.;
               for (int j = 0; j < N; ++j)
                 s1 += (pow(j+1, i+1)+beta)*(pow(x[j]/(j+1.), i+1) - 1.);
               s2 += s1*s1;
             }
             f = s2;
             break;
           }

    case PERM0: {// 5
            double beta = 10.;
            double s2 = 0.;
            for (int i = 0; i < N; ++i) {
              double s1 = 0.;
              for (int j = 0; j < N; ++j)
                s1 += (j+1.+beta)*(pow(x[j], i+1) - 1./pow(j+1, i+1));
              s2 += s1*s1;
            }
            f = s2;
            break;
          }

    case RASTRIGIN: {// 6
              double s = 0.;
              for (int i = 0; i < N; ++i)
                s += x[i]*x[i] - 10.*cos(2.*M_PI*x[i]);
              f = 10.*N+s;
              break;
            }

    case ROSENBROCK: {// 7
               double s = 0.;
               if( N==1 )
                 s=pow(x[0]-1., 2);
               else
                 for (int i = 0; i < N-1; ++i)
                   s += 100.*pow(x[i+1]-x[i]*x[i], 2) + pow(x[i]-1., 2);

               f = s;
               break;
             }

    case ROTATED_HYPER_ELLIPSOID: {// 8
                    double s = 0.;
                    double tmp=0.;
                    for (int i = 0; i < N; i++){
                      tmp += x[i];
                      s += pow(tmp,2);
                      }
                      f = s;
                      break;
                    }

    case SCHWEFEL: { // 9
               double s = 0.;
               for (int i = 0; i < N; ++i)
                 s += x[i]*sin(sqrt(fabs(x[i])));
               f = 418.9829*N-s;
               break;
             }

    case SPHERE: {// 10
             double s = 0.;
             for (int i = 0; i < N; ++i)
               s += x[i]*x[i];
             f = s;
             break;
           }

    case STYBLINSKI_TANG: { // 11
                  double s = 0.;
                  for (int i = 0; i < N; ++i)
                    s += pow(x[i], 4) - 16.*x[i]*x[i] + 5.*x[i];
                  f = 39.16599*N + .5*s;
                  break;
                }

    case SUM_OF_POWER: { // 12
                 double s = 0.;
                 for (int i = 0; i < N; ++i)
                   s += pow(fabs(x[i]), i+2);
                 f = s;
                 break;
               }

    case SUM_OF_SQUARES: { // 13
                 double s = 0.;
                 for (int i = 0; i < N; ++i)
                   s += (i+1.)*x[i]*x[i];
                 f = s;
                 break;
               }

    case ZAKHAROV: {// 14
               double s1 = 0., s2 = 0.;
               for (int i = 0; i < N; ++i) {
                 s1 += x[i]*x[i];
                 s2 += .5*(i+1.)*x[i];
               }
               f = s1 + pow(s2, 2) + pow(s2, 4);
               break;
             }

    default:
             printf("Function %d not found. Exiting.\n", rnd);
             exit(1);
  }
  //std::cout << "feval = " << f << std::endl;
  //return f;  /* our CMA maximizes (there's another "-" in the code) */
  if (std::isnan(f) || std::isinf(f)) { printf("Nan function\n"); abort(); }
  //std::cout << f << std::endl;
  /*
     std::ostringstream o;
     o << "[";
     for (int int i=0; i<dim; i++) {
     o << x[i];
     if (i < dim-1) o << " ";
     }
     o << "]";
     printf("Evaluated function in %s = %e\n", o.str().c_str(), f);
     */
  *output = f;
}

void get_upper_lower_bounds(double* const lower_bound, double* const upper_bound,
                int N, int * const info)
{
  int rnd = info[0] % _COUNT;  /* this defines which function to use */

  switch (rnd) {
    case ACKLEY: {
             for (int i = 0; i < N; ++i) {
               lower_bound[i] = -32.768;
               upper_bound[i] =  32.768;
             }
             break;
           }

    case DIXON_PRICE: {
                for (int i = 0; i < N; ++i) {
                  lower_bound[i] = -10;
                  upper_bound[i] =  10;
                }
                break;
              }

    case GRIEWANK: {
               for (int i = 0; i < N; ++i) {
                 lower_bound[i] = -600;
                 upper_bound[i] =  600;
               }
               break;
             }

    case LEVY: {
             for (int i = 0; i < N; ++i) {
               lower_bound[i] = -10;
               upper_bound[i] =  10;
             }
             break;
           }

    case PERM: {
             for (int i = 0; i < N; ++i) {
               lower_bound[i] = -N;
               upper_bound[i] =  N;
             }
             break;
           }

    case PERM0: {
            for (int i = 0; i < N; ++i) {
              lower_bound[i] = -N;
              upper_bound[i] =  N;
            }
            break;
          }

    case RASTRIGIN: {
              for (int i = 0; i < N; ++i) {
                lower_bound[i] = -5.12;
                upper_bound[i] =  5.12;
              }
              break;
            }

    case ROSENBROCK: {
               for (int i = 0; i < N; ++i) {
                 //restricted bounds because of poor perf
                 // bounds from https://www.sfu.ca/~ssurjano/rosen.html
                 lower_bound[i] = -2.048;
                 upper_bound[i] =  2.048;
                 //lower_bound[i] = -5;
                 //upper_bound[i] =  10;
               }
               break;
             }

    case ROTATED_HYPER_ELLIPSOID: {
                      for (int i = 0; i < N; ++i) {
                        lower_bound[i] = -65.536;
                        upper_bound[i] =  65.536;
                      }
                      break;
                    }

    case SCHWEFEL: {
               for (int i = 0; i < N; ++i) {
                 lower_bound[i] = -500;
                 upper_bound[i] =  500;
               }
               break;
             }

    case SPHERE: {
             for (int i = 0; i < N; ++i) {
               lower_bound[i] = -5.12;
               upper_bound[i] =  5.12;
             }
             break;
           }

    case STYBLINSKI_TANG: {
                  for (int i = 0; i < N; ++i) {
                    lower_bound[i] = -5;
                    upper_bound[i] =  5;
                  }
                  break;
                }

    case SUM_OF_POWER: {
                 for (int i = 0; i < N; ++i) {
                   lower_bound[i] = -1;
                   upper_bound[i] =  1;
                 }
                 break;
               }

    case SUM_OF_SQUARES: {
                 for (int i = 0; i < N; ++i) {
                   lower_bound[i] = -10;
                   upper_bound[i] =  10;
                 }
                 break;
               }

    case ZAKHAROV: {
               for (int i = 0; i < N; ++i) {
                 lower_bound[i] = -5;
                 upper_bound[i] =  10;
               }
               break;
             }

    default:
             printf("Function %d not found. Exiting.\n", rnd);
             exit(1);
  }
}

double eval_distance_from_optimum(const double* const found_optimum,
    int N, int* const info)
{
  double dist = 0.;
  int rnd = info[0] % _COUNT;  /* this defines which function to use */

  switch (rnd) {
    case ACKLEY: {
             for (int i = 0; i < N; ++i) {
               dist += pow(found_optimum[i], 2);
             }
             break;
           }

    case DIXON_PRICE: {
                for (int i = 0; i < N; ++i) {
                  dist += pow(found_optimum[i] - pow(2, -1.+1./pow(2,i)), 2);
                }
                break;
              }

    case GRIEWANK: {
               for (int i = 0; i < N; ++i) {
                 dist += pow(found_optimum[i], 2);
               }
               break;
             }

    case LEVY: {
             for (int i = 0; i < N; ++i) {
               dist += pow(found_optimum[i] -1., 2);
             }
             break;
           }

    case PERM: {
             for (int i = 0; i < N; ++i) {
               dist += pow(found_optimum[i] -i-1, 2);
             }
             break;
           }

    case PERM0: {
            for (int i = 0; i < N; ++i) {
              dist += pow(found_optimum[i] -1./(i+1), 2);
            }
            break;
          }

    case RASTRIGIN: {
              for (int i = 0; i < N; ++i) {
                dist += pow(found_optimum[i], 2);
              }
              break;
            }

    case ROSENBROCK: {
               for (int i = 0; i < N; ++i) {
                 dist += pow(found_optimum[i] -1., 2);
               }
               break;
             }

    case ROTATED_HYPER_ELLIPSOID: {
                      for (int i = 0; i < N; ++i) {
                        dist += pow(found_optimum[i], 2);
                      }
                      break;
                    }

    case SCHWEFEL: {
               for (int i = 0; i < N; ++i) {
                 dist += pow(found_optimum[i] -420.9687, 2);
               }
               break;
             }

    case SPHERE: {
             for (int i = 0; i < N; ++i) {
               dist += pow(found_optimum[i], 2);
             }
             break;
           }

    case STYBLINSKI_TANG: {
                  for (int i = 0; i < N; ++i) {
                    dist += pow(found_optimum[i] +2.903534, 2);
                  }
                  break;
                }

    case SUM_OF_POWER: {
                 for (int i = 0; i < N; ++i) {
                   dist += pow(found_optimum[i], 2);
                 }
                 break;
               }

    case SUM_OF_SQUARES: {
                 for (int i = 0; i < N; ++i) {
                   dist += pow(found_optimum[i], 2);
                 }
                 break;
               }

    case ZAKHAROV: {
               for (int i = 0; i < N; ++i) {
                 dist += pow(found_optimum[i], 2);
               }
               break;
             }

    default: {
           printf("Function %d not found. Exiting.\n", rnd);
           exit(1);
         }
  }

  //std::cout << sqrt(dist)  << std::endl;
  return sqrt(dist/N);
}
