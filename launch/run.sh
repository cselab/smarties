#!/bin/bash
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
unset LSB_AFFINITY_HOSTFILE #euler cluster
export MPICH_MAX_THREAD_SAFETY=multiple #MPICH
export MV2_ENABLE_AFFINITY=0 #MVAPICH
NPROCESS=$1
NTHREADS=$2
TASKPERN=$3
NMASTERS=$4

SETTINGSNAME=settings.sh
if [ ! -f $SETTINGSNAME ];then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source $SETTINGSNAME
if [ -x appSettings.sh ]; then
  source appSettings.sh
fi

SETTINGS+=" --nThreads ${NTHREADS}"
SETTINGS+=" --nMasters ${NMASTERS}"
SETTINGS+=" --ppn ${TASKPERN}"
export OMP_NUM_THREADS=${NTHREADS}
export OMP_PROC_BIND=CLOSE
export OMP_PLACES=cores
#export OMP_WAIT_POLICY=active
export OMP_MAX_TASK_PRIORITY=1
#export OMP_DISPLAY_ENV=TRUE
export OMP_DYNAMIC=FALSE

#echo $SETTINGS > settings.txt
env > environment.log
#echo ${NPROCESS} ${NTHREADS} $TASKPERN $NMASTERS

# Mpi call depends on whether user has open mpi or mpich, whether they are on
# a mac (which does not expose thread affinity), or on a linux cluster ...
# Let's assume for now users can sort this out themselves
isOpenMPI=$(mpirun --version | grep "Open MPI" | wc -l)
HOST=`hostname`
OS_D=`uname`
if [ ${isOpenMPI} -ge 1 ]; then
if [ ${OS_D} == 'Darwin' ] ; then
mpirun -n ${NPROCESS} ./rl ${SETTINGS} | tee out.log
else
mpirun -n ${NPROCESS} --map-by socket:PE=${NTHREADS} -report-bindings --mca mpi_cuda_support 0 ./rl ${SETTINGS} | tee out.log
fi
else # mpich / mvapich
#mpirun -n ${NPROCESS} -ppn ${TASKPERN} -bind-to core:${NTHREADS} valgrind --num-callers=100  --tool=memcheck  ./rl ${SETTINGS} | tee out.log
mpirun -n ${NPROCESS} -ppn ${TASKPERN} -bind-to core:${NTHREADS} ./rl ${SETTINGS} | tee out.log
fi

