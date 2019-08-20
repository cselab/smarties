SHELL := /bin/bash
CURPWD := $(shell pwd)
UNAME_S := $(shell uname -s)
config ?= prod
extra ?=

CPPFLAGS=
LIBS=

ifeq ($(UNAME_S),Darwin)
include make.macos
endif

ifeq "$(findstring eu-,$(shell hostname))" "eu-"
include make.euler
endif
ifeq "$(findstring euler,$(shell hostname))" "euler"
include make.euler
endif

ifeq "$(findstring daint,$(shell hostname))" "daint"
include make.daint_srun
endif

ifeq "$(findstring falcon,$(shell hostname))" "falcon"
include make.falcon
endif
ifeq "$(findstring panda,$(shell hostname))" "panda"
include make.falcon
endif

ifeq "$(findstring lair,$(shell hostname))" "lair"
include make.lair
endif

#ifeq "$(MPICXX)" ""
#CPPFLAGS+= -I$(CURPWD)/../extern/build/mpich-3.3/include/
#LIBS+= -L$(CURPWD)/../extern/build/mpich-3.3/lib/
#MPICXX=$(CURPWD)/../extern/build/mpich-3.3/bin/mpic++
#LD=$(CURPWD)/../extern/build/mpich-3.3/bin/mpic++
#else
LD=$(MPICXX)
#endif

ifeq "$(shell $(MPICXX) --version | grep -ci 'icpc')" "0"
ifeq "$(shell $(MPICXX) --version | grep -ci 'llvm')" "0"
compiler=gnu
else
compiler=llvm
endif
else
compiler=intel
endif

testdiff ?= off
ifeq "$(testdiff)" "on"
CPPFLAGS+= -D__CHECK_DIFF
endif

precision ?= double
app_precision ?= $(precision)
ifeq "$(precision)" "single"
CPPFLAGS+= -DSINGLE_PREC
endif

ifeq "$(compiler)" "intel"
include make.icc.flags
else
ifeq "$(compiler)" "llvm"
include make.llvm.flags
else
include make.gcc.flags
endif
endif

DIRS = $(sort $(dir $(wildcard ../source/*/)))

OBJ_FILES = \
	PPO.o \
	NAF.o \
	DPG.o \
	DQN.o \
	ACER.o \
	RACER.o \
	CMALearner.o \
	Learner_pytorch.o \
	Learner_approximator.o \
	AlgoFactory.o \
	Sequences.o \
	Learner.o \
	Sampling.o \
	Collector.o \
	Math_tests.o \
	MemoryBuffer.o \
	DataCoordinator.o \
	MemoryProcessing.o \
	Builder.o \
	Network.o \
	Optimizer.o \
	CMA_Optimizer.o \
	StatsTracker.o \
	Communicator.o \
	Approximator.o \
	Launcher.o \
	Settings.o \
	Profiler.o \
	Master.o \
	Worker.o \
	Engine.o

CPP_FILES = $(notdir $(OBJ_FILES:.o=.cpp))

.DEFAULT_GOAL := all

###################### Compile Communicator Library ###########################
PYBINDINC:=$(shell python3 -m pybind11 --includes) $(NOWARNP11)
PYBINDSUF:=$(shell python3-config --extension-suffix)
ifeq ($(UNAME_S),Darwin)
PYBINDLD= $(LDFLAGS) -undefined dynamic_lookup
else
PYBINDLD= $(LDFLAGS)
endif

PATH_COMM_PY11=../source/smarties_pybind11.cpp
smarties_pybind11.o:
	$(MPICXX) $(CPPFLAGS) $(PYBINDINC) -fPIC -c $(PATH_COMM_PY11) -o $@

lib: $(OBJ_FILES)
	mkdir -p ../lib
	$(MPICXX) -shared -fPIC $(LDFLAGS) -o ../lib/libsmarties.so $^ $(LIBS)
	@echo "Remember to update the environmental variables to include the smarties library like:"
	@cd ../include && echo "echo 'export SMARTIES_INCLUDE=`pwd`' >> ~/.bash_profile"
	@cd ../lib && echo "echo 'export SMARTIES_LIB=`pwd`' >> ~/.bash_profile"
	@cd ../ && echo "echo 'export SMARTIES_ROOT=`pwd`' >> ~/.bash_profile"

#echo "echo 'export LD_LIBRARY_PATH=\"\$${LD_LIBRARY_PATH}:`pwd`\"' >> ~/.bash_profile"

py11: $(OBJ_FILES) smarties_pybind11.o
	mkdir -p ../lib
	$(MPICXX) -shared $(PYBINDLD) -o ../lib/smarties$(PYBINDSUF) $^ $(LIBS)


#$(MPICXX) $(CPPFLAGS) -DSMARTIES_LIB -shared -fPIC \
#	$(PATH_COMM).cpp -o libsmarties.so
#g++ -shared -Wl,-soname,$(LIBNAME) -o $(LIBNAME) CommunicatorLib.o

#exec: $(OBJ_FILES) main.o
#	$(LD) $(LDFLAGS) -o rl main.o $(LIBS) $(OBJ_FILES)

exec: lib main.o
	$(LD) $(LDFLAGS) -o rl main.o -L../lib/ -lsmarties

all: lib

main.o: ../source/main.cpp
	$(MPICXX) $(extra) $(CPPFLAGS) -c $< -o $@

%.o: %.cpp
	$(MPICXX) $(extra) $(CPPFLAGS) -fPIC -c $< -o $@

vpath %.cpp $(DIRS)

clean:
	rm -f *.o *.so *.s *.d ../lib/lib*
