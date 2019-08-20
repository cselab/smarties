export INTERNALAPP=false

if [[ "${SKIPMAKE}" != "true" ]] ; then
make -C ${SMARTIES_ROOT}/apps/cart_pole_many
fi

cp ${SMARTIES_ROOT}/apps/cart_pole_many/cart-pole ${RUNDIR}/exec


