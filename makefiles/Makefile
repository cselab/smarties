hn ?= $(shell hostname)
username ?= $(shell whoami)
UNAME_S := $(shell uname -s)
config ?= prod
extra ?=
compiler ?= gnu

CPPFLAGS=-D__Smarties_
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

exp ?= GAUS
ifeq "$(exp)" "GAUS"
CPPFLAGS+= -DADV_GAUS
endif
nexp ?= 1
CPPFLAGS+= -DNEXPERTS=$(nexp)

testdiff ?= off
ifeq "$(testdiff)" "on"
CPPFLAGS+= -D__CHECK_DIFF
endif

app ?= ext_app_test

ifeq "$(compiler)" "intel"
include make.icc.flags
else
include make.gcc.flags
endif

DIRS = $(sort $(dir $(wildcard ../source/*/)))
OBJ_FILES = ArgParser.o main.o ObjectFactory.o Communicator.o RACER.o \
		PPO.o ACER.o VRACER.o AtariEnvironment.o Approximator.o Network.o \
		Learner_onPolicy.o MemoryBuffer.o Learner_offPolicy.o Optimizer.o \
		Environment.o DQN.o RETPG.o Math_tests.o Communicator_internal.o \
		Learner.o NAF.o DPG.o Scheduler.o Sequences.o

CPP_FILES = $(notdir $(OBJ_FILES:.o=.cpp))

.DEFAULT_GOAL := rl

ifneq "$(MAKECMDGOALS)" "clean"
-include $(notdir $(patsubst %.cpp,%.d,$(CPP_FILES)))
endif

############### Compile application as static library if needed ###############
ifneq "$(app)" "ext_app_test"
# let's check if app requires some flag / lib
ifneq ("$(wildcard ../apps/$(app)/make.flags)","")
include ../apps/$(app)/make.flags
endif
CPPFLAGS+= -DINTERNALAPP # app is linked directly to smarties

lib: Communicator.o
	make -C ../apps/$(app)/ clean
	make -C ../apps/$(app)/ CC=$(CC)
else
# app is compiled independently and uses communication through sockets
# this is just for testing:
lib: ext_app_test.o Communicator.o
	ar rs libsimulation.a $^
endif
LIBS+= -L. -lsimulation

rl: $(OBJ_FILES) lib
	$(CC) $(OBJ_FILES) $(LIBS) -o $@

%.o: %.cpp
	$(CC) $(extra)  $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CC) $(extra)  $(CPPFLAGS) -c -MD $<

vpath %.cpp $(DIRS)
vpath %.h   $(DIRS)

clean:
	rm -f *.o *.so *.s *.d libsimulation.a rl
