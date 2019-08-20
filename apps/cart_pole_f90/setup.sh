#cp ../apps/test_mpi_cart_pole/runArguments* ${BASEPATH}${RUNFOLDER}/

#make -C ../makefiles/ clean

make -C ../makefiles/ app=test_f90_cart_pole -j #config=debug #compiler=intel #config=nan # #testdiff=on

# For debugging some utility of communicator:
#cat <<EOF >${BASEPATH}${RUNFOLDER}/appSettings.sh
#SETTINGS+=" --appSettings runArguments00.sh,runArguments01.sh,runArguments02.sh,runArguments03.sh "
#SETTINGS+=" --nStepPappSett 16384,16284,16184,16084 "
##SETTINGS+=" --nStepPappSett 4194304,2097152,1048576,0 "
#EOF
#chmod +x ${BASEPATH}${RUNFOLDER}/appSettings.sh
