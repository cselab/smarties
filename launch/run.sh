#!/bin/bash
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
HOST=`hostname`
OS_D=`uname`

unset LSB_AFFINITY_HOSTFILE #euler cluster
export MPICH_MAX_THREAD_SAFETY=multiple #MPICH
export MV2_ENABLE_AFFINITY=0 #MVAPICH
export OPENBLAS_NUM_THREADS=1

SETTINGSNAME=settings.sh
if [ ! -f $SETTINGSNAME ];then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source $SETTINGSNAME
if [ -x appSettings.sh ]; then
  source appSettings.sh
fi

SETTINGS+=" --nWorkers ${NWORKERS}"
SETTINGS+=" --nMasters ${NMASTERS}"
SETTINGS+=" --nThreads ${NTHREADS}"
export OMP_NUM_THREADS=${NTHREADS}
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores

env > environment.log

# Mpi call depends on whether user has open mpi or mpich, whether they are on
# a mac (which does not expose thread affinity), or on a linux cluster ...
# Let's assume for now users can sort this out themselves
isOpenMPI=$(mpirun --version | grep "Open MPI" | wc -l)
if [ ${isOpenMPI} -ge 1 ]; then
if [ ${OS_D} == 'Darwin' ] ; then
mpirun -n ${NPROCESS} ./rl ${SETTINGS} | tee out.log
else
mpirun -n ${NPROCESS} --map-by ppr:1:socket:pe=${NTHREADS}  ./rl ${SETTINGS} | tee out.log
fi
else # mpich / mvapich
#mpirun -n ${NPROCESS} -bind-to core:${NTHREADS} valgrind --num-callers=100  --tool=memcheck --leak-check=yes  --track-origins=yes ./rl ${SETTINGS} | tee out.log
mpirun -n ${NPROCESS} -bind-to core:${NTHREADS} ./rl ${SETTINGS} | tee out.log
fi
