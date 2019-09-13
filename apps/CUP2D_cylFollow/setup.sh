# compile executable:
COMPILEDIR=${SMARTIES_ROOT}/../CubismUP_2D/makefiles
if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ${COMPILEDIR} leadFollow -j4
fi

# copy executable:
cp ${COMPILEDIR}/leadFollow ${RUNDIR}/exec

# copy simulation settings files:
cp runArguments* ${RUNDIR}/

# command line args to find app-required settings, each to be used for fixed
# number of steps so as to increase sim fidelity as training progresses
export EXTRA_LINE_ARGS=" --nStepPappSett 1048576,524288,262144,0 \
--appSettings runArguments00.sh,runArguments01.sh,runArguments02.sh,runArguments03.sh "

# heavy application, needs dedicated processes
export MPI_RANKS_PER_ENV=1

#SETTINGS+=" --nStepPappSett 2097152,1048576,524288,0 "
#SETTINGS+=" --nStepPappSett 4194304,2097152,1048576,0 "
