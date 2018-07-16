#UNAME_S := $(shell uname -s)
#ifeq ($(UNAME_S),Darwin)
#CPP = g++-6
#CC = gcc-6
#else
CPP = g++
CC = gcc
#endif

SRC_DIR = ./
COM_DIR = ../../source/Communicators/
VPATH := $(SRC_DIR):$(COM_DIR)
WARNINGS = -Wall -Wextra -Wredundant-decls -Wshadow #-Woverloaded-virtual
WARNINGS+= -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings #-Wlogical-op
WARNINGS+= -Wmissing-declarations -Wno-unused-parameter -Wno-float-equal #i dont like float equals but... CMA-ES...

#CPPFLAGS+=-std=c++11 -Wall -O0 # -fopenmp
CPPFLAGS+= -std=c++11 -O0 $(WARNINGS) -I$(COM_DIR)# -fopenmp
CFLAGS+= -O0 $(WARNINGS)
LIBS= -lstdc++ -lm

OBJ_FILES_PP = engine_cmaes.o cmaes_learn.o fitfun.o Communicator.o
OBJ_FILES =    cmaes.o

.DEFAULT_GOAL := engine_cmaes

engine_cmaes: $(OBJ_FILES) $(OBJ_FILES_PP)
	$(CPP) $(LIBS) $^ -o $@ #-fopenmp

engine_cmaes.o: engine_cmaes.cpp
	$(CPP) $(CPPFLAGS) -MD -c $< -o $@

communicator.o: communicator.cpp
	$(CPP) $(CPPFLAGS) -MD -c $< -o $@

cmaes.o: cmaes.c
	$(CC) $(CFLAGS) -MD -c $< -o $@

cmaes_learn.o: cmaes_learn.cpp
	$(CPP) $(CPPFLAGS) -MD -c $< -o $@

fitfun.o: fitfun.c
	$(CPP) $(CPPFLAGS) -MD -c $< -o $@

clean:
	rm -f *.o
	rm -f *.d
	rm -f engine_cmaes
