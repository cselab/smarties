make -C ../apps/boatNav clean
make -C ../apps/boatNav

cp ../apps/boatNav/boatNav   ${RUNDIR}/exec
cp -r ../apps/boatNav ${RUNDIR}/srcDir

