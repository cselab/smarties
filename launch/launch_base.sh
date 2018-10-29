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
	echo "Usage: ./launch_base.sh RUNFOLDER APP SETTINGS_PATH (NWORKERS) (NTHREADS) (NMASTERS) (NNODES)"
	exit 1
fi
if [ $# -gt 3 ] ; then #n worker ranks
export NWORKERS=$4
else
export NWORKERS=1
fi
if [ $# -gt 4 ] ; then
export NMASTERS=$5
else
export NMASTERS=1 #n master ranks
fi
if [ $# -gt 5 ] ; then
export NPROCESS=$6
else
export NPROCESS=1 # total number of ranks
fi
if [ $# -gt 6 ] ; then #n threads on each master
export NTHREADS=$7
else
export NTHREADS=$([[ $(uname) = 'Darwin' ]] && sysctl -n hw.physicalcpu_max || lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)
fi

cp run.sh ${BASEPATH}${RUNFOLDER}/run.sh
cp ../makefiles/rl ${BASEPATH}${RUNFOLDER}/rl
cp $0 ${BASEPATH}${RUNFOLDER}/launch_smarties.sh
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
git log | head  > ${BASEPATH}${RUNFOLDER}/gitlog.log
git diff > ${BASEPATH}${RUNFOLDER}/gitdiff.log

cd ${BASEPATH}${RUNFOLDER}

HOST=`hostname`

if [ ${HOST:0:5} == 'euler' ] || [ ${HOST:0:3} == 'eu-' ] ;
#if [ ${HOST:0:5} == 'euler' ] #|| [ ${HOST:0:3} == 'eu-' ] ;
then
export NTHREADS=18
NPROCESSORS=$((${NPROCESS}*${NTHREADS}))
bsub -J ${RUNFOLDER} -R fullnode -R "rusage[mem=128]" -R "select[model==XeonGold_6150]" -n ${NPROCESSORS} -W 120:00 < run.sh
else
 source run.sh
fi
