CXX ?= g++

extra ?=
OBJ_FILES = test-fit.o
SRC_DIR = ./
COM_DIR = ../../source/
VPATH := $(SRC_DIR):$(COM_DIR)
CPPFLAGS+= -std=c++14 -Wall -O2 -I$(COM_DIR)
#CPPFLAGS+= -std=c++14 -Wall -O0 -g -I$(COM_DIR)
CPPFLAGS+= -Wextra -Wfloat-equal -Wundef -Wcast-align
CPPFLAGS+= -Wwrite-strings -Wlogical-op -Wmissing-declarations -Wredundant-decls
CPPFLAGS+= -Wshadow -Woverloaded-virtual -Wuninitialized -Wno-div-by-zero

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Darwin)
ifeq ($(CXX), c++)
#LD=g++-9
#CXX=g++-9
endif
endif

.DEFAULT_GOAL := test-fit

test-fit: $(OBJ_FILES)
	$(CXX) $(extra) -L../../makefiles/ -o $@ $^ -lsmarties

%.o: %.cpp
	$(CXX) $(extra) $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CXX) $(extra) $(CPPFLAGS) -MD -c $<

#vpath %.cpp $(DIRS)
#vpath %.h   $(DIRS)

clean:
	rm -f *.o *.s *.d test-fit
