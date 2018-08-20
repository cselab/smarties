#include <stdio.h>
#include <sched.h>
#include <omp.h>
#include <mpi.h>
#include <sys/unistd.h>
#include <sys/stat.h>
#include <signal.h>
#include <vector>
#include <array>

#include <cpuid.h>

#define CPUID(INFO, LEAF, SUBLEAF) __cpuid_count(LEAF, SUBLEAF, INFO[0], INFO[1], INFO[2], INFO[3])

#define GETCPU(CPU) {                                  \
        uint32_t CPUInfo[4];                           \
        CPUID(CPUInfo, 1, 0);                          \
        /* CPUInfo[1] is EBX, bits 24-31 are APIC ID */\
        if ( (CPUInfo[3] & (1 << 9)) == 0) {           \
          CPU = -1;  /* no APIC on chip */             \
        }                                              \
        else {                                         \
          CPU = (unsigned)CPUInfo[1] >> 24;            \
        }                                              \
        if (CPU < 0) CPU = 0;                          \
      }

using namespace std;
int main(int argc, char** argv)
{
  int provided;
  //const auto SECURITY = MPI_THREAD_MULTIPLE;
  const auto SECURITY = MPI_THREAD_SERIALIZED;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    printf("%d %d\n", provided, SECURITY);
  if (provided < SECURITY) {
    printf("The MPI implementation does not have required thread support\n");
    return 1;
  }
  vector<array<double,3>> partial_forces = vector<array<double,3>>(omp_get_max_threads(), {0.,0.,0.});
  printf("%g\n",partial_forces[0][0]);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  //char hostname[1024];
  //hostname[1023] = '\0';
  //gethostname(hostname, 1023);
  #pragma omp parallel
  {
      int thread_num = omp_get_thread_num();
      int cpu_num;
      GETCPU(cpu_num);
      printf("Rank %d Thread %3d on CPU %d\n",
      world_rank, thread_num, cpu_num);
  }

  MPI_Finalize();
  return 0;
}
