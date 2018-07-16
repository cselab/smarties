
cp ../apps/test_rnn/launch.sh ${BASEPATH}${RUNFOLDER}/launchSim.sh
cp ../apps/test_rnn/testApp.py     ${BASEPATH}${RUNFOLDER}/
cp ../source/Communicators/Communicator.py ${BASEPATH}${RUNFOLDER}/

cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
SETTINGS+=" --nnType GRU"
EOF
chmod +x ${BASEPATH}${RUNFOLDER}/appSettings.sh
