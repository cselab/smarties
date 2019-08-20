export INTERNALAPP=true

# compile executable:
if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ${SMARTIES_ROOT}/../CubismUP_2D/makefiles smartCyl -j4
fi

# copy executable:
cp ${SMARTIES_ROOT}/../CubismUP_2D/makefiles/smartCyl ${RUNDIR}/exec

# copy simulation settings files:
cp ${SMARTIES_ROOT}/apps/CUP2D_smartCyl/runArguments* ${RUNDIR}/

# write file for launch_base.sh to read app-required settings:
cat <<EOF >${RUNDIR}/appSettings.sh
SETTINGS+=" --appSettings runArguments01.sh "
SETTINGS+=" --nStepPappSett 0 "
EOF
chmod 755 ${RUNDIR}/appSettings.sh

