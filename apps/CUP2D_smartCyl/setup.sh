export INTERNALAPP=true

# compile executable:
COMPILEDIR=${SMARTIES_ROOT}/../CubismUP_2D/makefiles
if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ${COMPILEDIR} smartCyl -j4
fi

# copy executable:
cp ${COMPILEDIR}/smartCyl ${RUNDIR}/exec

# copy simulation settings files:
cp runArguments* ${RUNDIR}/

# command line args to find app-required settings, each to be used for fixed
# number of steps so as to increase sim fidelity as training progresses
export EXTRA_LINE_ARGS=" --nStepPappSett 0 --appSettings runArguments00.sh "

# heavy application, needs dedicated processes
export MPI_RANKS_PER_ENV=1
