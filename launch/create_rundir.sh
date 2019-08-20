#!/bin/bash
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#

RUNNAME=$1

HOST=`hostname`
if [ ${HOST:0:5} == 'euler' ] || [ ${HOST:0:3} == 'eu-' ] ; then
export BASEPATH="${SCRATCH}/smarties/"
elif [ ${HOST:0:5} == 'daint' ] ; then
export BASEPATH="${SCRATCH}/smarties/"
else
export BASEPATH="../runs/"
fi

export RUNDIR=${BASEPATH}${RUNNAME}
mkdir -p ${RUNDIR}
