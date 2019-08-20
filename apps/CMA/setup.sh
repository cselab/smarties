export INTERNALAPP=false
make -C ../apps/CMA/ clean
make -C ../apps/CMA/

cp ../apps/CMA/engine_cmaes ${BASEPATH}${RUNFOLDER}/
cp ../apps/CMA/cmaes_initials.par ${BASEPATH}${RUNFOLDER}/
cp ../apps/CMA/cmaes_signals.par ${BASEPATH}${RUNFOLDER}/
cp ../apps/CMA/launch.sh ${BASEPATH}${RUNFOLDER}/launchSim.sh
