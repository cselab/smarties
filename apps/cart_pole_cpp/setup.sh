export INTERNALAPP=false

if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ${SMARTIES_ROOT}/apps/cart_pole_cpp
fi

cp ${SMARTIES_ROOT}/apps/cart_pole_cpp/cart-pole ${RUNDIR}/exec
