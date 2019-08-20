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
RESTARTDIR=${SMARTIES_ROOT}/apps/CUP2D_cylFollow/glider_timeopt_rho2_noise005
cp ${RESTARTDIR}/agent* ${RUNDIR}/

# write file for launch_base.sh to read app-required settings:
cat <<EOF >${RUNDIR}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh"
SETTINGS+=" --nStepPappSett 0 "
SETTINGS+=" --restart . "
SETTINGS+=" --bTrain 0 "
EOF
chmod 755 ${RUNDIR}/appSettings.sh
