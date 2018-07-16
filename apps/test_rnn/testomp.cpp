#include <iostream>
#include <cmath>
#include <cstdio>
#include "omp.h"

int main (int argc, char** argv)
{
printf("%d\n", omp_get_max_threads());
return 0;
}
