CXX = mpic++


#LDFLAGS+= -fopenmp
LIBS+= -L${SMARTIES_ROOT}/lib -lsmarties
CPPFLAGS+= -std=c++14 -Wall -O2 -I${SMARTIES_ROOT}/include
#CPPFLAGS+= -std=c++14 -Wall -g -O0 -I${SMARTIES_ROOT}/include
CPPFLAGS+= -Wextra -Wfloat-equal -Wundef -Wcast-align -Wno-unused-parameter
CPPFLAGS+= -Wwrite-strings -Wlogical-op -Wmissing-declarations -Wredundant-decls
CPPFLAGS+= -Wshadow -Woverloaded-virtual -Wuninitialized -Wno-div-by-zero

.DEFAULT_GOAL := cart-pole

cart-pole: cart-pole.o
	$(CXX) $(extra) -o $@ $^ ${LIBS}

%.o: %.cpp
	$(CXX) $(extra) $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CXX) $(extra) $(CPPFLAGS) -MD -c $<

clean:
	rm -f *.o *.s *.d cart-pole
