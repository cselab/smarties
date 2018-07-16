#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
ENV=$2
TASK=$3
SETTINGSNAME=$4
BDAINTMC=0
if [ $BDAINTMC -eq 1 ] ; then
CONSTRAINT=mc
else
CONSTRAINT=gpu
fi

MYNAME=`whoami`
BASEPATH="/scratch/snx3000/${MYNAME}/smarties/"

mkdir -p ${BASEPATH}${RUNFOLDER}
ulimit -c unlimited

if [ $# -gt 4 ] ; then
NSLAVESPERMASTER=$5
else
NSLAVESPERMASTER=1 #n tasks per node
fi

if [ $# -gt 5 ] ; then
NMASTERS=$6
else
if [ $BDAINTMC -eq 1 ] ; then
NMASTERS=2 #n master ranks
else
NMASTERS=1 #n master ranks
fi
fi
if [ $# -gt 6 ] ; then
NNODES=$7
else
NNODES=1 #threads per master
fi

if [ $# -gt 7 ] ; then
NTHREADS=$8
else
if [ $BDAINTMC -eq 1 ] ; then
NTHREADS=18 #n master ranks
else
NTHREADS=12 #n master ranks
fi
fi

NTASKPERMASTER=$((1+${NSLAVESPERMASTER})) # master plus its slaves
NPROCESS=$((${NMASTERS}*$NTASKPERMASTER))
NTASKPERNODE=$((${NPROCESS}/${NNODES}))

cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
LD_PRELOAD=${HOME}/glew-2.1.0/install/lib64/libGLEW.so python3 ../Communicator_dmc.py \$1 $ENV $TASK
EOF

#this handles app-side setup
cp ../../source/Communicators/Communicator.py     ${BASEPATH}${RUNFOLDER}/
cp ../../source/Communicators/Communicator_gym.py ${BASEPATH}${RUNFOLDER}/
chmod +x ${BASEPATH}${RUNFOLDER}/launchSim.sh

cp ../../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/exec
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/policy_settings.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch.sh
git log | head  > ${BASEPATH}${RUNFOLDER}/gitlog.log
git diff > ${BASEPATH}${RUNFOLDER}/gitdiff.log

cd ${BASEPATH}${RUNFOLDER}
if [ ! -f settings.sh ];then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source settings.sh
SETTINGS+=" --nMasters ${NMASTERS}"
SETTINGS+=" --nThreads ${NTHREADS}"
SETTINGS+=" --ppn ${NTASKPERNODE}"
echo $SETTINGS > settings.txt
echo ${SETTINGS}

cat <<EOF >daint_sbatch
#!/bin/bash -l

#SBATCH --account=s658
#SBATCH --job-name="${RUNFOLDER}"
#SBATCH --output=${RUNFOLDER}_out_%j.txt
#SBATCH --error=${RUNFOLDER}_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=${NMASTERS}
#SBATCH --ntasks-per-node=${NTASKPERNODE}
#SBATCH --constraint=${CONSTRAINT}

# #SBATCH --partition=debug
# #SBATCH --time=00:30:00
# #SBATCH --mail-user="${MYNAME}@ethz.ch"
# #SBATCH --mail-type=ALL

export OMP_NUM_THREADS=${NTHREADS}
export CRAY_CUDA_MPS=1
export OMP_PROC_BIND=CLOSE
export OMP_PLACES=cores

srun --ntasks ${NPROCESS} --threads-per-core=2 --cpu_bind=sockets --cpus-per-task=${NTHREADS} --ntasks-per-node=${NTASKPERNODE} ./exec ${SETTINGS}
EOF

chmod 755 daint_sbatch

sbatch daint_sbatch
cd -
