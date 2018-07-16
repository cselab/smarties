SOCK=$1
#module load valgrind
#valgrind --tool=memcheck --leak-check=full  --vgdb=yes --track-origins=yes --show-reachable=no --show-possibly-lost=no  ../engine_cmaes ${SOCK} ${NTHREADS}
../engine_cmaes ${SOCK}
