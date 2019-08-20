#!/bin/bash
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
RUNFOLDER=$1
APP=$2 # NoFrameskip-v4 will be added  at the end of the task name

if [ $# -lt 2 ] ; then
echo "Usage: ./launch_atari.sh RUNFOLDER GAMEID( NoFrameskip-v4 will be added internally ) (... optional arguments defined in launch_base.sh )"
exit 1
fi

source create_rundir.sh

export INTERNALAPP=false
export EXECNAME="exec.py $APP"

cp ../apps/OpenAI_gym_atari/exec.py ${BASEPATH}${RUNFOLDER}/

source launch_base.sh
