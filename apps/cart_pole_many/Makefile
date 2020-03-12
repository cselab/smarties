CXX=mpic++
ifneq ($(wildcard ${SMARTIES_ROOT}/extern/.*),)
CXX=${SMARTIES_ROOT}/extern/bin/mpic++
endif

LIBS+= -L${SMARTIES_ROOT}/lib -lsmarties
CPPFLAGS+= -std=c++14 -Wall -O2 -I${SMARTIES_ROOT}/include

CPPFLAGS+= -Wextra -Wfloat-equal -Wundef -Wcast-align
CPPFLAGS+= -Wwrite-strings -Wlogical-op -Wmissing-declarations -Wredundant-decls
CPPFLAGS+= -Wshadow -Woverloaded-virtual -Wuninitialized -Wno-div-by-zero

.DEFAULT_GOAL := cart-pole

cart-pole: cart-pole.o
	$(CXX) $(extra) -o $@ $^ ${LIBS}

%.o: %.cpp
	$(CXX) $(extra)  $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CXX) $(extra)  $(CPPFLAGS) -MD -c $<

#vpath %.cpp $(DIRS)
#vpath %.h   $(DIRS)

clean:
	rm -f *.o *.s *.d cart-pole
