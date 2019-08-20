export INTERNALAPP=true

# compile executable, assumes that smarties and CUP2D are in the same directory:
if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ${SMARTIES_ROOT}/../CubismUP_2D/makefiles blowfish -j4
fi

# copy executable:
cp ${SMARTIES_ROOT}/../CubismUP_2D/makefiles/blowfish ${RUNDIR}/exec

# copy simulation settings files:
cp ${SMARTIES_ROOT}/apps/CUP2D_blowfish/runArguments* ${RUNDIR}/

# write file for launch_base.sh to read app-required settings:
cat <<EOF >${RUNDIR}/appSettings.sh
SETTINGS+=" --appSettings runArguments00.sh,runArguments01.sh,runArguments02.sh,runArguments03.sh "
SETTINGS+=" --nStepPappSett 262144,262144,262144,0 "
EOF
chmod 755 ${RUNDIR}/appSettings.sh

#SETTINGS+=" --nStepPappSett 2097152,1048576,524288,0 "
#export LD_LIBRARY_PATH=/cluster/home/novatig/VTK-7.1.0/Build/lib/:$LD_LIBRARY_PATH
#cp ${HOME}/CubismUP_2D/makefiles/blowfish ${BASEPATH}${RUNFOLDER}/
