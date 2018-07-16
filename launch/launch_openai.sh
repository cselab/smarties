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
APP=$2
SETTINGSNAME=$3

if [ $# -lt 3 ] ; then
echo "Usage: ./launch_openai.sh RUNFOLDER APP SETTINGS_PATH (for other optional params see launch_base.sh)"
exit 1
fi

source create_rundir.sh

HOSTNAME=`hostname`
if [ ${HOSTNAME:0:5} == 'falco' ] || [ ${HOSTNAME:0:5} == 'panda' ]
then
cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
/home/novatig/Python-3.5.2/build/bin/python3.5 ../Communicator_gym.py \$1 $APP
EOF
else
cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
python3 ../Communicator_gym.py \$1 $APP
EOF
fi
chmod +x ${BASEPATH}${RUNFOLDER}/launchSim.sh

cp ../source/Communicators/Communicator.py     ${BASEPATH}${RUNFOLDER}/
cp ../source/Communicators/Communicator_gym.py ${BASEPATH}${RUNFOLDER}/

source launch_base.sh

#python ../openaibot.py \$1 $APP
#xvfb-run -s "-screen $DISPLAY 1400x900x24" -- python ../openaibot.py \$1 $APP
#vglrun -c proxy python3 ../Communicator.py \$1 $APP
