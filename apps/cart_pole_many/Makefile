CC = g++

LIBS+= -L${SMARTIES_ROOT}/lib -lsmarties
CPPFLAGS+= -std=c++14 -Wall -O2 -I${SMARTIES_ROOT}/include

CPPFLAGS+= -Wextra -Wfloat-equal -Wundef -Wcast-align
CPPFLAGS+= -Wwrite-strings -Wlogical-op -Wmissing-declarations -Wredundant-decls
CPPFLAGS+= -Wshadow -Woverloaded-virtual -Wuninitialized -Wno-div-by-zero

.DEFAULT_GOAL := cart-pole

cart-pole: cart-pole.o
	$(CC) $(extra) -o $@ $^ ${LIBS}

%.o: %.cpp
	$(CC) $(extra)  $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CC) $(extra)  $(CPPFLAGS) -MD -c $<

#vpath %.cpp $(DIRS)
#vpath %.h   $(DIRS)

clean:
	rm -f *.o *.s *.d cart-pole
