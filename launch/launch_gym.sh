#!/bin/bash
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
RUNFOLDER=$1
APP=$2

if [ $# -lt 2 ] ; then
echo "Usage: ./launch_gym.sh RUNFOLDER ENVIRONMENT_APP (... optional arguments defined in launch_base.sh )"
exit 1
fi

source create_rundir.sh

export INTERNALAPP=false
export EXECNAME="exec.py $APP"

cp ../apps/OpenAI_gym/exec.py ${BASEPATH}${RUNFOLDER}/exec.py
#UNCOMMENT FOR EVALUATING LEARNED POLICY, TODO CLEANUP
#cp ../apps/OpenAI_gym/exec_eval.py ${BASEPATH}${RUNFOLDER}/exec.py
#export INTERNALAPP=true

source launch_base.sh

#python ../openaibot.py \$1 $APP
#xvfb-run -s "-screen $DISPLAY 1400x900x24" -- python ../openaibot.py \$1 $APP
#vglrun -c proxy python3 ../Communicator.py \$1 $APP
