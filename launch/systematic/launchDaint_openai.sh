#!/bin/bash
EXECNAME=rl
RUNFOLDER=$1
APP=$2
SETTINGSNAME=$3
BDAINTMC=0
if [ $BDAINTMC -eq 1 ] ; then
CONSTRAINT=mc
else
CONSTRAINT=gpu
fi

MYNAME=`whoami`
BASEPATH="/scratch/snx3000/${MYNAME}/smarties/"
#BASEPATH="/scratch/snx1600/${MYNAME}/smarties/"
mkdir -p ${BASEPATH}${RUNFOLDER}
ulimit -c unlimited

if [ $# -gt 3 ] ; then
NSLAVESPERMASTER=$4
else
NSLAVESPERMASTER=1 #n tasks per node
fi

if [ $# -gt 4 ] ; then
NMASTERS=$5
else
if [ $BDAINTMC -eq 1 ] ; then
#NMASTERS=2 #n master ranks
NMASTERS=1 #n master ranks
else
NMASTERS=1 #n master ranks
fi
fi
if [ $# -gt 5 ] ; then
NNODES=$6
else
NNODES=1 #threads per master
fi

if [ $# -gt 6 ] ; then
NTHREADS=$7
else
if [ $BDAINTMC -eq 1 ] ; then
#NTHREADS=18 #n master ranks
NTHREADS=36 #n master ranks
else
NTHREADS=12 #n master ranks
fi
fi


NTASKPERMASTER=$((1+${NSLAVESPERMASTER})) # master plus its slaves
NPROCESS=$((${NMASTERS}*$NTASKPERMASTER))
NTASKPERNODE=$((${NPROCESS}/${NNODES}))

cat <<EOF >${BASEPATH}${RUNFOLDER}/launchSim.sh
python3 ../Communicator_gym.py \$1 $APP
EOF

cat <<EOF >${BASEPATH}${RUNFOLDER}/factory
Environment exec=../launchSim.sh n=1
EOF
chmod +x ${BASEPATH}${RUNFOLDER}/launchSim.sh

cp ../../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/exec
cp ../../source/Communicators/Communicator.py     ${BASEPATH}${RUNFOLDER}/
cp ../../source/Communicators/Communicator_gym.py ${BASEPATH}${RUNFOLDER}/
cp ${SETTINGSNAME} ${BASEPATH}${RUNFOLDER}/settings.sh
cp $0 ${BASEPATH}${RUNFOLDER}/launch.sh
git log | head  > ${BASEPATH}${RUNFOLDER}/gitlog.log
git diff HEAD > ${BASEPATH}${RUNFOLDER}/gitdiff.log

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
# #SBATCH --account=eth2
#SBATCH --job-name="${RUNFOLDER}"
#SBATCH --output=${RUNFOLDER}_out_%j.txt
#SBATCH --error=${RUNFOLDER}_err_%j.txt
#SBATCH --nodes=${NNODES}
#SBATCH --ntasks-per-node=${NTASKPERNODE}
#SBATCH --constraint=${CONSTRAINT}

#SBATCH --time=16:00:00
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


