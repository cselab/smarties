F90 = mpif90
CXX = mpic++

F90FLAGS+= -Wall -Wextra -Wconversion -pedantic -O2 #-g -O0
LIBS+= -lgfortran -lmpifort

#LDFLAGS+= -fopenmp
LIBS+= -L${SMARTIES_ROOT}/lib -lsmarties
CPPFLAGS+= -std=c++14 -Wall -O2 -I${SMARTIES_ROOT}/include
#CPPFLAGS+= -std=c++14 -Wall -g -O0 -I${SMARTIES_ROOT}/include
CPPFLAGS+= -Wextra -Wfloat-equal -Wundef -Wcast-align -Wno-unused-parameter
CPPFLAGS+= -Wwrite-strings -Wlogical-op -Wmissing-declarations -Wredundant-decls
CPPFLAGS+= -Wshadow -Woverloaded-virtual -Wuninitialized -Wno-div-by-zero

#.DEFAULT_GOAL := cart_pole

.PHONY: all clean

all: post-build

pre-build:
ifeq (,$(wildcard ./smarties.f90))
	ln -s $(SMARTIES_ROOT)/include/smarties.f90
endif

post-build: main-build
	@echo "Done."

main-build: pre-build
	@$(MAKE) --no-print-directory cart_pole


cart_pole: main.o app_main.o cart_pole.o smarties.o
	$(CXX) $(extra) -o $@ $^ ${LIBS}

app_main.o: cart_pole.o smarties.o

%.o: %.cpp
	$(CXX) $(extra) $(CPPFLAGS) -c $< -o $@

%.o: %.f90
	$(F90) $(F90FLAGS) -c $< -o $@

%.d: %.cpp
	$(CXX) $(extra) $(CPPFLAGS) -MD -c $<

clean:
	rm -f *.o *.s *.d *.mod smarties.f90 cart_pole
