#!/bin/bash
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
RUNFOLDER=$1
ENV=$2
TASK=$3

if [ $# -lt 3 ] ; then
echo "Usage: ./launch_dmcs.sh RUNFOLDER ENV TASK (... optional arguments defined in launch_base.sh )"
exit 1
fi

source create_rundir.sh

export INTERNALAPP=false
export EXECNAME="exec.py $ENV $TASK"
export DISABLE_MUJOCO_RENDERING=1

cp ../apps/Deepmind_control/exec.py ${BASEPATH}${RUNFOLDER}/

shift 2 # hack because for deepmind we need two args to describe env
./launch_base.sh $RUNFOLDER $@



