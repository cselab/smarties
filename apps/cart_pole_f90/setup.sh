export INTERNALAPP=true

if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ${SMARTIES_ROOT}/apps/cart_pole_f90
fi

cp ${SMARTIES_ROOT}/apps/cart_pole_f90/cart_pole ${RUNDIR}/exec
