cp ../apps/smartCyl/runArguments* ${BASEPATH}${RUNFOLDER}/

#make -C ../makefiles/ clean
rm ../makefiles/libsimulation.a
rm ../makefiles/rl
make -C ../makefiles/ app=smartCyl precision=single -j4
#config=segf
cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --appSettings runArguments05.sh,runArguments00.sh,runArguments01.sh"
SETTINGS+=" --nStepPappSett 2097152,2097152,0 "
EOF
chmod 755 ${BASEPATH}${RUNFOLDER}/appSettings.sh
