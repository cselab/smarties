CXX ?= g++
ifneq ($(wildcard ${SMARTIES_ROOT}/extern/.*),)
CXX=${SMARTIES_ROOT}/extern/bin/mpic++
endif

OBJ_FILES = boatNav.o odeSolve.o
LIBS+= -L${SMARTIES_ROOT}/lib -lsmarties
CPPFLAGS+= -std=c++14 -Wall -O2 -I${SMARTIES_ROOT}/include
#CPPFLAGS+=-std=c++14 -Wall -O0 -I${SMARTIES_ROOT}/include

CPPFLAGS+= -Wextra -Wfloat-equal -Wundef -Wcast-align
CPPFLAGS+= -Wwrite-strings -Wlogical-op -Wmissing-declarations -Wredundant-decls
CPPFLAGS+= -Wshadow -Woverloaded-virtual -Wuninitialized -Wno-div-by-zero

#-I/usr/local/Cellar/python/3.6.1/Frameworks/Python.framework/Versions/3.6/include/python3.6m/
#LIBS = -L/usr/local/Cellar/python/3.6.1/Frameworks/Python.framework/Versions/3.6/lib/python3.6/config-3.6m-darwin/ -lpython3.6m

#CPPFLAGS+= -fsanitize=address
#LIBS += -fsanitize=address

.DEFAULT_GOAL := boatNav

boatNav: $(OBJ_FILES)
	$(CXX) $(extra) $(LIBS) -o $@ $^ ${LIBS}

%.o: %.cpp
	$(CXX) $(extra) $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CXX) $(extra) $(CPPFLAGS) -MD -c $<

#vpath %.cpp $(DIRS)
#vpath %.h   $(DIRS)

clean:
	rm -f *.o
	rm -f *.s
	rm -f *.d
	rm -f boatNav
