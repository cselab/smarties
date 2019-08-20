export INTERNALAPP=true

# compile executable:
if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ${SMARTIES_ROOT}/../CubismUP_2D/makefiles leadFollow -j4
fi

# copy executable:
cp ${SMARTIES_ROOT}/../CubismUP_2D/makefiles/leadFollow ${RUNDIR}/exec

# copy simulation settings files:
cp ${SMARTIES_ROOT}/apps/CUP2D_2fish/runArguments* ${RUNDIR}/

# write file for launch_base.sh to read app-required settings:
cat <<EOF >${RUNDIR}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh,runArguments01.sh,runArguments02.sh,runArguments03.sh "
SETTINGS+=" --nStepPappSett 1048576,524288,262144,0 "
EOF
chmod 755 ${RUNDIR}/appSettings.sh


