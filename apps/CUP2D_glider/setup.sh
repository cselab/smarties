export INTERNALAPP=true

# compile executable:
COMPILEDIR=${SMARTIES_ROOT}/../CubismUP_2D/makefiles
if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ${COMPILEDIR} glider -j4
fi

# copy executable:
cp ${COMPILEDIR}/glider ${RUNDIR}/exec

# write simulation settings files:
cat <<EOF >${RUNDIR}/runArguments00.sh
../launchSim.sh -CFL 0.1 -DLM 1 -lambda 1e5 -iterativePenalization 1 \
-poissonType cosine -muteAll 1 -bpdx 32 -bpdy 32 -tdump 1 -nu 0.0004 -tend 0 \
-shapes 'glider_semiAxisX=.125_semiAxisY=.025_rhoS=2_xpos=.6_ypos=.4_bFixed=1_bForced=0'
EOF

#copy restart files:
#cp glider_timeopt_rho2_noise005/agent* ${RUNDIR}/

# command line args to find app-required settings, each to be used for fixed
# number of steps so as to increase sim fidelity as training progresses
export EXTRA_LINE_ARGS=" --nStepPappSett 0 --restart . --bTrain 0 \
--appSettings runArguments00.sh "

# heavy application, benefits from dedicated processes, here just eval:
export MPI_RANKS_PER_ENV=0
#export EXECNAME=exec
