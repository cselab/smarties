#!/bin/bash
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
EXECNAME=rl
RUNFOLDER=$1
ENV=$2
TASK=$3
SETTINGSNAME=$4

if [ $# -lt 4 ] ; then
echo "Usage: ./launch_deepmind.sh RUNFOLDER ENV TASK SETTINGS_PATH (for other optional params see launch_base.sh)"
exit 1
fi

source create_rundir.sh

HOSTNAME=`hostname`
if [ ${HOSTNAME:0:5} == 'falco' ] || [ ${HOSTNAME:0:5} == 'panda' ]
then
cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
LD_PRELOAD=libstdc++.so.6 ${HOME}/Python-3.5.2/build/bin/python3.5 ../Communicator_dmc.py \$1 $ENV $TASK
EOF
else
cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
python3 ../Communicator_dmc.py \$1 $ENV $TASK
EOF
fi
chmod +x ${BASEPATH}${RUNFOLDER}/launchSim.sh

cp ../source/Communicators/Communicator.py     ${BASEPATH}${RUNFOLDER}/
cp ../source/Communicators/Communicator_dmc.py ${BASEPATH}${RUNFOLDER}/

export DISABLE_MUJOCO_RENDERING=1
shift 2 # hack because for deepmind we need to args to describe env
./launch_base.sh $RUNFOLDER $@
