#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
NTHREADS=$2
APP=$3
SETTINGSNAME=$4

if [ $# -lt 4 ] ; then
	echo "Usage: ./launch_openai.sh RUNFOLDER OMP_THREADS APP SETTINGS_PATH (POLICY_PATH) (N_MPI_TASK_PER_NODE)"
	exit 1
fi
if [ $# -gt 4 ] ; then
NSLAVESPERMASTER=$5
else
NSLAVESPERMASTER=1 #n master ranks
fi
if [ $# -gt 5 ] ; then
NMASTERS=$6
else
NMASTERS=1 #n master ranks
fi
if [ $# -gt 6 ] ; then
NNODES=$7
else
NNODES=1 #n master ranks
fi

NTASKPERMASTER=$((1+${NSLAVESPERMASTER})) # master plus its slaves
NPROCESS=$((${NMASTERS}*$NTASKPERMASTER))
NTASKPERNODE=$((${NPROCESS}/${NNODES}))

MYNAME=`whoami`
HOST=`hostname`
#if [ ${HOST:0:5} == 'euler' ] || [ ${HOST:0:5} == 'eu-lo' ] ; then
#if [ ${HOST:0:5} == 'euler' ] || [ ${HOST:0:5} == 'eu-lo' ] || [ ${HOST:0:4} == 'eu-c' ] ; then
#	BASEPATH="/cluster/scratch/${MYNAME}/smarties/"
#else
	BASEPATH="../runs/"
#fi
mkdir -p ${BASEPATH}${RUNFOLDER}
if [ $# -gt 7 ] ; then
echo Restart from path $8
if [ -d $8 ]; then
cp $8/agent*_net_1stMom.raw ${BASEPATH}${RUNFOLDER}/
cp $8/agent*_net_2ndMom.raw ${BASEPATH}${RUNFOLDER}/
cp $8/agent*_net_tgt_weights.raw ${BASEPATH}${RUNFOLDER}/
cp $8/agent*_net_weights.raw ${BASEPATH}${RUNFOLDER}/
cp $8/agent_0[0-9]_scaling.raw ${BASEPATH}${RUNFOLDER}/
else
echo Directory not found.
exit 1
fi
fi

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

cp run.sh ${BASEPATH}${RUNFOLDER}/run.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch_smarties.sh
cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/rl
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/policy_settings.sh
git log | head  > ${BASEPATH}${RUNFOLDER}/gitlog.log
git diff > ${BASEPATH}${RUNFOLDER}/gitdiff.log

cd ${BASEPATH}${RUNFOLDER}

NPROCESS=$((${NNODES}*${NTASKPERNODE})) # || [ ${HOST:0:4} == 'eu-c' ] 
#if [ ${HOST:0:5} == 'euler' ] || [ ${HOST:0:5} == 'eu-lo' ] ; then
#if [ ${HOST:0:5} == 'euler' ] || [ ${HOST:0:5} == 'eu-lo' ] || [ ${HOST:0:4} == 'eu-c' ] ; then
#	NTHREADSPERNODE=24
#	NPROCESSORS=$((${NNODES}*${NTHREADSPERNODE}))
#	bsub -J ${RUNFOLDER} -R "select[model==XeonE5_2680v3]" -n ${NPROCESSORS} -W 24:00 ./run.sh ${NPROCESS} ${NTHREADS} ${NTASKPERNODE} 1
#else
./run.sh ${NPROCESS} ${NTHREADS} ${NTASKPERNODE} ${NMASTERS}
#fi
