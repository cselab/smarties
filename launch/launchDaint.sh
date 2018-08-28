#!/bin/bash
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
RUNFOLDER=$1
NNODES=$2
APP=$3
SETTINGSNAME=$4

if [ $# -lt 4 ] ; then
	echo "Usage: ./launch_openai.sh RUNFOLDER MPI_NODES APP SETTINGS_PATH (POLICY_PATH) (N_MPI_TASK_PER_NODE OMP_THREADS)"
	exit 1
fi

MYNAME=`whoami`
BASEPATH="/scratch/snx3000/${MYNAME}/smarties/"
mkdir -p ${BASEPATH}${RUNFOLDER}
#ulimit -c unlimited
#lfs setstripe -c 1 ${BASEPATH}${RUNFOLDER}
NMASTERS=1
NTASKPERNODE=1
NTHREADS=12
NPROCESS=$((${NTASKPERNODE}*${NNODES}))

#this handles app-side setup (incl. copying the factory)
#this must handle all app-side setup (as well as copying the factory)
if [ -d ${APP} ]; then
	if [ -x ${APP}/setup.sh ] ; then
		source ${APP}/setup.sh ${BASEPATH}${RUNFOLDER}
	else
		echo "${APP}/setup.sh does not exist or I cannot execute it"
		exit 1
	fi
else
	if [ -x ../apps/${APP}/setup.sh ] ; then
		source ../apps/${APP}/setup.sh ${BASEPATH}${RUNFOLDER}
	else
		echo "../apps/${APP}/setup.sh does not exist or I cannot execute it"
		exit 1
	fi
fi

cp ../makefiles/rl ${BASEPATH}${RUNFOLDER}/rl
if [ ! -x ../makefiles/rl ] ; then
	echo "../makefiles/rl not found! - exiting"
	exit 1
fi
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/policy_settings.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch_smarties.sh
git log | head  > ${BASEPATH}${RUNFOLDER}/gitlog.log
git diff > ${BASEPATH}${RUNFOLDER}/gitdiff.log

cd ${BASEPATH}${RUNFOLDER}
if [ ! -f settings.sh ] ; then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit 1
fi
source settings.sh
if [ -x appSettings.sh ]; then
source appSettings.sh
fi
SETTINGS+=" --nThreads ${NTHREADS}"
SETTINGS+=" --nMasters ${NMASTERS}"
SETTINGS+=" --ppn ${NTASKPERNODE}"
echo $SETTINGS > settings.txt
echo ${SETTINGS}

cat <<EOF >daint_sbatch
#!/bin/bash -l

#SBATCH --account=s658
#SBATCH --job-name="${RUNFOLDER}"
#SBATCH --output=${RUNFOLDER}_out_%j.txt
#SBATCH --error=${RUNFOLDER}_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=${NNODES}
#SBATCH --ntasks-per-node=${NTASKPERNODE}
#SBATCH --constraint=gpu

# #SBATCH --time=00:30:00
# #SBATCH --partition=debug
# #SBATCH --constraint=mc

#SBATCH --mail-user="${MYNAME}@ethz.ch"
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=${NTHREADS}
export CRAY_CUDA_MPS=1
export OMP_PROC_BIND=CLOSE
export OMP_PLACES=cores

srun --ntasks ${NPROCESS} --ntasks-per-node=${NTASKPERNODE} --cpus-per-task=${NTHREADS} --threads-per-core=1 ./rl ${SETTINGS}
EOF

chmod 755 daint_sbatch
sbatch daint_sbatch
