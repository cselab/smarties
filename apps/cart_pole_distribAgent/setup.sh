make

cp cart-pole ${RUNDIR}/exec
NCARTS=2
export MPI_RANKS_PER_ENV=${NCARTS}
#export EXTRA_LINE_ARGS=${NCARTS}
