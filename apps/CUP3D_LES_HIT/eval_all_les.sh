export SKIPMAKE=true

export LES_RL_NETTYPE=${LES_RL_NETTYPE:-FFNN}
export LES_RL_N_TSIM=${LES_RL_N_TSIM:-100}
export LES_RL_NBLOCK=${LES_RL_NBLOCK:-4}

if [ ${LES_RL_NBLOCK} == 8 ] ; then
blocksize=4
elif [ ${LES_RL_NBLOCK} == 4 ] ; then
blocksize=8
elif [ ${LES_RL_NBLOCK} == 2 ] ; then
blocksize=16
else
echo "ERROR"
exit 1
fi

make -C ~/CubismUP_3D/makefiles/ clean
make -C ~/CubismUP_3D/makefiles/ hdf=false bs=${blocksize} accfft=false -j rlHIT

THISDIR=${SMARTIES_ROOT}/apps/CUP3D_LES_HIT

for REW in GERMANO ; do
for GRIDAGENT in 0 ; do
#for ACT in 2 4 8 16 ; do
for ACT in 4 8 ; do
for REN in 065 076 088 103 120 140 163 ; do
#for RE in RE065 RE076 RE088 RE103 RE120 RE140 RE163 ; do
#for REN in 060 065 070 076 082 088 095 103 111 120 130 140 151 163 176 190 205 ; do
#for REN in 060 070 082 095 111 130 151 176 190 205 ; do
RE=RE${REN}

export LES_RL_EVALUATE=${RE}
export LES_RL_FREQ_A=${ACT}
export LES_RL_GRIDACT=1 #${GRIDAGENT}

SPEC=${REW}_${LES_RL_NETTYPE}_${LES_RL_NBLOCK}blocks_act`printf %02d $LES_RL_FREQ_A`

if [ ${GRIDAGENT} == 0 ] ; then
BASENAME=GridAgents_BlockRestart
RESTART=BlockAgent
export LES_RL_GRIDACTSETTINGS=0
else
BASENAME=GridAgents_GridRestart
RESTART=GridAgent
export LES_RL_GRIDACTSETTINGS=1
fi

RUNDIR=${BASENAME}_00_${SPEC}_sim${LES_RL_N_TSIM}_${RE}
RESTARTDIR=${THISDIR}/trained_${RESTART}_${SPEC}/
echo $RUNDIR

if [ -d ${SMARTIES_ROOT}/runs/${RUNDIR} ] ; then
echo ${RUNDIR} exists
else
smarties.py CUP3D_LES_HIT -n 1 -r ${RUNDIR} --restart ${RESTARTDIR} --nEvalEpisodes 1 --clockHours 1 --nTaskPerNode 2
#echo ${RUNDIR} todo
fi 

#--printAppStdout

done
done
done
done
