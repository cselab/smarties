#!/bin/bash
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
RUNFOLDER=$1
SETTINGSNAME=$3

if [ $# -lt 3 ] ; then
	echo "Usage: ./launch_base.sh RUNFOLDER APP SETTINGS_PATH (NSLAVESPERMASTER) (NTHREADS) (NMASTERS) (NNODES)"
	exit 1
fi
if [ $# -gt 3 ] ; then #n worker ranks per each master
NSLAVESPERMASTER=$4
else
NSLAVESPERMASTER=1
fi
if [ $# -gt 4 ] ; then #n threads on each master
NTHREADS=$5
else
NTHREADS=$([[ $(uname) = 'Darwin' ]] && sysctl -n hw.physicalcpu_max || lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)
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

cp run.sh ${BASEPATH}${RUNFOLDER}/run.sh
cp ../makefiles/rl ${BASEPATH}${RUNFOLDER}/rl
cp $0 ${BASEPATH}${RUNFOLDER}/launch_smarties.sh
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
git log | head  > ${BASEPATH}${RUNFOLDER}/gitlog.log
git diff > ${BASEPATH}${RUNFOLDER}/gitdiff.log

cd ${BASEPATH}${RUNFOLDER}

HOST=`hostname`
# || [ ${HOST:0:4} == 'eu-c' ]
if [ ${HOST:0:5} == 'euler' ] || [ ${HOST:0:5} == 'eu-lo' ] ; then
	NTHREADSPERNODE=24
	NPROCESSORS=$((${NNODES}*${NTHREADSPERNODE}))
	bsub -J ${RUNFOLDER} -R "select[model==XeonE5_2680v3]" -n ${NPROCESSORS} -W 24:00 ./run.sh ${NPROCESS} ${NTHREADS} ${NTASKPERNODE} 1
else
./run.sh ${NPROCESS} ${NTHREADS} ${NTASKPERNODE} ${NMASTERS}
fi
