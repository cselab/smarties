CPP = g++
CC = gcc
ifneq ($(wildcard ${SMARTIES_ROOT}/extern/.*),)
CXX=${SMARTIES_ROOT}/extern/bin/mpic++
CC=${SMARTIES_ROOT}/extern/bin/mpicc
endif

WARNINGS = -Wall -Wextra -Wredundant-decls -Wshadow #-Woverloaded-virtual
WARNINGS+= -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings #-Wlogical-op
WARNINGS+= -Wmissing-declarations -Wno-unused-parameter -Wno-float-equal #i dont like float equals but... CMA-ES...

LIBS+= -L${SMARTIES_ROOT}/lib -lsmarties
CPPFLAGS+= -std=c++14 $(WARNINGS) -O2 -I${SMARTIES_ROOT}/include

CFLAGS+= -O0 $(WARNINGS)
LIBS+= -lstdc++ -lm

OBJ_FILES_PP = engine_cmaes.o cmaes_learn.o fitfun.o Communicator.o
OBJ_FILES =    cmaes.o

.DEFAULT_GOAL := engine_cmaes

engine_cmaes: $(OBJ_FILES) $(OBJ_FILES_PP)
	$(CXX) $(LIBS) $^ -o $@ #-fopenmp

engine_cmaes.o: engine_cmaes.cpp
	$(CXX) $(CPPFLAGS) -MD -c $< -o $@

communicator.o: communicator.cpp
	$(CXX) $(CPPFLAGS) -MD -c $< -o $@

cmaes.o: cmaes.c
	$(CC) $(CFLAGS) -MD -c $< -o $@

cmaes_learn.o: cmaes_learn.cpp
	$(CXX) $(CPPFLAGS) -MD -c $< -o $@

fitfun.o: fitfun.c
	$(CXX) $(CPPFLAGS) -MD -c $< -o $@

clean:
	rm -f *.o
	rm -f *.d
	rm -f engine_cmaes
