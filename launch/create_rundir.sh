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

if [ $# -lt 3 ] ; then
echo "Usage: ./launch_openai.sh RUNFOLDER APP SETTINGS_PATH (for other optional params see launch_base.sh)"
exit 1
fi

MYNAME=`whoami`
HOST=`hostname`
#if [ ${HOST:0:5} == 'euler' ] || [ ${HOST:0:3} == 'eu-' ] ; then
#	export BASEPATH="/cluster/scratch/${MYNAME}/smarties/"
#else
export BASEPATH="../runs/"
#fi
mkdir -p ${BASEPATH}${RUNFOLDER}
