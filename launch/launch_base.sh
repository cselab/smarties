#!/bin/bash
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
RUNNAME=$1

if [ $# -lt 2 ] ; then
	echo "Usage: ./launch_base.sh RUNNAME ENVIRONMENT_APP (SETTINGS_PATH default is 'settings/VRACER.json' ) (NTHREADS default read from system, but unrealiable on clusters) (NNODES default 1) (NMASTERS default 1) (NWORKERS default 1)"
	exit 1
fi
HOST=`hostname`

if [ $# -gt 2 ] ; then
  if [ -f ${3} ]; then
    export SETTINGSNAME=$3
  else
    export SETTINGSNAME=${SMARTIES_ROOT}/settings/$3
  fi
else export SETTINGSNAME=${SMARTIES_ROOT}/settings/VRACER.json
fi
if [ ! -f ${SETTINGSNAME} ]; then
echo "Could not find settings file" ${SETTINGSNAME}
exit 1
fi
################################################################################
########### FIRST, read cores per node and available number of nodes ###########
################################################################################
if [ $# -gt 3 ] ; then export NTHREADS=$4
else
################################################################################
if [ ${HOST:0:5} == 'euler' ] || [ ${HOST:0:3} == 'eu-' ] ; then
export NTHREADS=36
elif [ ${HOST:0:5} == 'daint' ] ; then
export NTHREADS=12
else
export NTHREADS=$([[ $(uname) = 'Darwin' ]] && sysctl -n hw.physicalcpu_max || lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)
fi
################################################################################
fi

if [ $# -gt 4 ] ; then export NNODES=$5
else export NNODES=1
fi

################################################################################
######## SECOND, depending on environment type, specify distribution of ########
####### resources between learning (master) ranks and env (worker) ranks #######
################################################################################

################################################################################
if [[ "${INTERNALAPP}" == "true" ]] ; then # WORKERS RUN ON DEDICATED MPI RANKS
################################################################################

# if distribution not specified, assume we want as many workers as possible
if [ $# -gt 5 ] ; then export NMASTERS=$6
else export NMASTERS=1
fi
if [ $# -gt 6 ] ; then export NWORKERS=$7
elif [ $NNODES -gt 1 ] ; then export NWORKERS=$(( $NNODES - $NMASTERS ))
else export NWORKERS=1
fi

# BOTH MASTERS AND WORKERS ARE CREATED DURING INITIAL MPIRUN CALL:
NPROCESSES=$(( $NWORKERS + $NMASTERS ))

################################################################################
else # THEN WORKERS ARE FORKED PROCESSES RUNNING ON SAME CORES AS MASTERS
################################################################################

# if distribution not specified, assume we want as many masters as possible:
if [ $# -gt 5 ] ; then export NMASTERS=$6
else export NMASTERS=$NNODES
fi
# assume we fork one env process per master mpi rank:
if [ $# -gt 6 ] ; then export NWORKERS=$7
else export NWORKERS=$NNODES
fi

# ONLY MASTERS ARE INCLUDED AMONG PROCESSES IN INITIAL MPIRUN CALL:
NPROCESSES=$NMASTERS

################################################################################
fi # END OF LOGIC ON ${INTERNALAPP} AND RESOURCES DISTRIBUTION
################################################################################

# Compute number of processes running on each node:
ZERO=$(( $NPROCESSES % $NNODES ))
if [ $ZERO != 0 ] ; then
echo "ERROR: unable to map NWORKERS and NMASTERS onto NNODES"
exit 1
fi
export NPROCESSPERNODE=$(( $NPROCESSES / $NNODES ))
echo "NWORKERS:"$NWORKERS "NMASTERS:"$NMASTERS "NNODES:"$NNODES "NPROCESSPERNODE:"$NPROCESSPERNODE

################################################################################
############################## PREPARE RUNFOLDER ###############################
################################################################################
if [ -z "$SMARTIES_ROOT" ] ; then
echo "ERROR: Environment variable SMARTIES_ROOT is not set. Read the README.rst"
exit 1
fi
if [ ! -f ${SMARTIES_ROOT}/lib/libsmarties.so ]; then
echo "ERROR: smarties library not found."
exit 1
fi
if [ ! -f ${SMARTIES_ROOT}/lib/smarties.cpython-* ]; then
echo "ERROR: pybind11 smarties library not found."
exit 1
fi

EXECPATH=`echo $EXECNAME | cut -f1 -d" "`
if [ ! -x ${RUNDIR}/${EXECPATH} ]; then
echo "ERROR: Application executable not found! Revise app's setup.sh"
exit 1
fi

cp -f $0 ${RUNDIR}/launch_smarties.sh
cp -f ${SETTINGSNAME} ${RUNDIR}/settings.json
git log | head  > ${RUNDIR}/gitlog.log
git diff > ${RUNDIR}/gitdiff.log

################################################################################
############################### PREPARE HPC ENV ################################
################################################################################
unset LSB_AFFINITY_HOSTFILE #euler cluster
export MPICH_MAX_THREAD_SAFETY=multiple #MPICH
export MV2_ENABLE_AFFINITY=0 #MVAPICH
export OMP_NUM_THREADS=${NTHREADS}
export OPENBLAS_NUM_THREADS=1
export CRAY_CUDA_MPS=1
export PYTHONPATH=${PYTHONPATH}:${SMARTIES_ROOT}/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SMARTIES_ROOT}/lib
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${SMARTIES_ROOT}/lib
#export PATH=`pwd`/../extern/build/mpich-3.3/bin/:$PATH
#export LD_LIBRARY_PATH=`pwd`/../extern/build/mpich-3.3/lib/:$LD_LIBRARY_PATH
################################################################################

cd ${RUNDIR}

################################################################################
############################ READ SMARTIES SETTINGS ############################
################################################################################
if [ -x appSettings.sh ]; then source appSettings.sh ; fi
SETTINGS+=" --nWorkers ${NWORKERS}"
SETTINGS+=" --nMasters ${NMASTERS}"
SETTINGS+=" --nThreads ${NTHREADS}"
if [ "${INTERNALAPP}" == "true" ] ; then SETTINGS+=" --runInternalApp 1"
else SETTINGS+=" --runInternalApp 0"
fi

################################################################################
#################################### EULER #####################################
################################################################################
if [ ${HOST:0:5} == 'euler' ] || [ ${HOST:0:3} == 'eu-' ] ; then

# override trick to run without calling bsub:
if [ "${RUNLOCAL}" == "true" ] ; then
mpirun -n ${NPROCESSES} --map-by ppr:${NPROCESSPERNODE}:node \
  ./${EXECNAME} ${SETTINGS} | tee out.log
fi

WCLOCK=${WCLOCK:-24:00}
# compute the number of CPU CORES to ask euler:
export NPROCESSORS=$(( ${NNODES} * ${NTHREADS} ))

bsub -n ${NPROCESSORS} -J ${RUNFOLDER} \
  -R "select[model==XeonGold_6150] span[ptile=${NTHREADS}]" -W ${WCLOCK} \
  mpirun -n ${NPROCESSES} --map-by ppr:${NPROCESSPERNODE}:node \
  ./${EXECNAME} ${SETTINGS} | tee out.log

################################################################################
#################################### DAINT #####################################
################################################################################
elif [ ${HOST:0:5} == 'daint' ] ; then

WCLOCK=${WCLOCK:-24:00:00}
# did we allocate a node?
srun hostname &> /dev/null
if [[ "$?" -gt "0" ]] ; then # no we did not. call sbatch:

cat <<EOF >daint_sbatch
#!/bin/bash -l
#SBATCH --account=s929 --job-name="${RUNFOLDER}" --time=${WCLOCK}
#SBATCH --output=${RUNFOLDER}_out_%j.txt --error=${RUNFOLDER}_err_%j.txt
#SBATCH --nodes=${NNODES} --constraint=gpu
srun -n ${NPROCESSES} --nodes=${NNODES}  --ntasks-per-node=${NPROCESSPERNODE} \
  ./${EXECNAME} ${SETTINGS}
EOF

chmod 755 daint_sbatch
sbatch daint_sbatch

else

srun -n ${NPROCESSES} --nodes ${NNODES} --ntasks-per-node ${NPROCESSPERNODE} \
  ./${EXECNAME} ${SETTINGS}

fi

################################################################################
############################## LOCAL WORKSTATION ###############################
################################################################################
else

mpirun -n ${NPROCESSES} --map-by ppr:${NPROCESSPERNODE}:node \
  ./${EXECNAME} ${SETTINGS} | tee out.log
#mpirun -n ${NPROCESSES} --map-by ppr:${NPROCESSPERNODE}:node \
#  valgrind --num-callers=100  --tool=memcheck --leak-check=yes \
#  --track-origins=yes --show-reachable=yes \
#  ./rl ${SETTINGS}

fi
