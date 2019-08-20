export INTERNALAPP=true

if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ${SMARTIES_ROOT}/apps/cart_pole_distribAgent
fi

cp ${SMARTIES_ROOT}/apps/cart_pole_distribAgent/cart-pole ${RUNDIR}/exec
