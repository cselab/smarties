export SKIPMAKE=true

# how many cases to consider
#for nblocks in 2 4 8; do
for nblocks in 4; do

if [ ${nblocks} == 8 ] ; then
blocksize=4
elif [ ${nblocks} == 4 ] ; then
blocksize=8
elif [ ${nblocks} == 2 ] ; then
blocksize=16
else
echo "ERROR"
exit 1
fi

make -C ~/CubismUP_3D/makefiles/ clean
#make -C ~/CubismUP_3D/makefiles/ hdf=false bs=${blocksize} accfft=false -j rlHIT
make -C ~/CubismUP_3D/makefiles/  bs=${blocksize} -j rlHIT

for run in 16 17; do

export LES_RL_NBLOCK=$nblocks
export LES_RL_N_TSIM=20
POSTNAME=sim${LES_RL_N_TSIM}_RUN${run}
SPEC=NAI

# several options for actuation freq (relative to kolmogorov time)
# bcz it affects run time we allocate different number of resources:

################################################################################
export LES_RL_GRIDACT=0
export LES_RL_NETTYPE=GRU
BASENAME=BlockAgents_${SPEC}_${LES_RL_NETTYPE}_${nblocks}blocks
echo $BASENAME
################################################################################

export LES_RL_FREQ_A=1
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
#smarties.py CUP3D_LES_HIT -n 16 -r ${RUNDIR}

export LES_RL_FREQ_A=2
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
#smarties.py CUP3D_LES_HIT  -n 11 -r ${RUNDIR}

export LES_RL_FREQ_A=4
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
#smarties.py CUP3D_LES_HIT -n 6 -r ${RUNDIR}

export LES_RL_FREQ_A=8
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
#smarties.py CUP3D_LES_HIT -n 4 -r ${RUNDIR}

export LES_RL_FREQ_A=16
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
#smarties.py CUP3D_LES_HIT -n 4 -r ${RUNDIR}


################################################################################
export LES_RL_GRIDACT=0
export LES_RL_NETTYPE=FFNN
BASENAME=BlockAgents_${SPEC}_${LES_RL_NETTYPE}_${nblocks}blocks
echo $BASENAME
################################################################################

export LES_RL_FREQ_A=1
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
#smarties.py CUP3D_LES_HIT -n 25 -r ${RUNDIR}

export LES_RL_FREQ_A=2
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
#smarties.py CUP3D_LES_HIT -n 12 -r ${RUNDIR}

export LES_RL_FREQ_A=4
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
#smarties.py CUP3D_LES_HIT -n 7 -r ${RUNDIR}

export LES_RL_FREQ_A=8
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
smarties.py CUP3D_LES_HIT -n 4 -r ${RUNDIR}

export LES_RL_FREQ_A=16
RUNDIR=${BASENAME}_act`printf %02d $LES_RL_FREQ_A`_${POSTNAME}
#smarties.py CUP3D_LES_HIT -n 4 -r ${RUNDIR}

done
done


