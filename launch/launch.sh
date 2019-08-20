#!/bin/bash
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#  Created by Guido Novati (novatig@ethz.ch).
#

RUNNAME=$1
APP=$2

if [ $# -lt 2 ] ; then
echo "Usage: ./launch_gym.sh RUNNAME ENVIRONMENT_APP (... optional arguments defined in launch_base.sh )"
exit 1
fi

source create_rundir.sh

#variable that allows setup.sh to change executable name:
export EXECNAME=exec

#this must handle all app-side setup (as well as copying the factory)
if [ -d ${APP} ]; then
	if [ -x ${APP}/setup.sh ] ; then
		source ${APP}/setup.sh ${RUNDIR}
	else
		echo "${APP}/setup.sh does not exist or I cannot execute it"
		exit 1
	fi
else
	if [ -x ../apps/${APP}/setup.sh ] ; then
		source ../apps/${APP}/setup.sh ${RUNDIR}
	else
		echo "../apps/${APP}/setup.sh does not exist or I cannot execute it"
		exit 1
	fi
fi

source launch_base.sh
