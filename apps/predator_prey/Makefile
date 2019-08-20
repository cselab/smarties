CC = g++

LIBS+= -L${SMARTIES_LIB} -lsmarties
CPPFLAGS+= -std=c++14 -Wall -O2 -I${SMARTIES_INCLUDE}

onepred ?= false
ifneq "$(onepred)" "false"
OBJ_FILES = pp.o
else
OBJ_FILES = twoPredOnePrey.o
endif

CPPFLAGS+= -Wextra -Wfloat-equal -Wundef -Wcast-align
CPPFLAGS+= -Wwrite-strings -Wlogical-op -Wmissing-declarations -Wredundant-decls
CPPFLAGS+= -Wshadow -Woverloaded-virtual -Wuninitialized -Wno-div-by-zero

ifeq "$(findstring eu-,$(shell hostname))" "eu-"
CPPFLAGS+= -I ${PYTHON_ROOT}/include/python2.7
LIBS += -L ${PYTHON_ROOT}/lib64 -lpython2.7
else
CPPFLAGS+= -I/usr/include/python2.7
LIBS += -lpython2.7
endif

.DEFAULT_GOAL := pp

pp: $(OBJ_FILES)
	$(CC) $(extra) $(LIBS) $^ -o $@

%.o: %.cpp
	$(CC) $(extra)  $(CPPFLAGS) -c $< -o $@

%.d: %.cpp
	$(CC) $(extra)  $(CPPFLAGS) -MD -c $<

#vpath %.cpp $(DIRS)
#vpath %.h   $(DIRS)

clean:
	rm -f *.o
	rm -f *.s
	rm -f *.d
	rm -f pp
