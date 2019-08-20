.DEFAULT_GOAL := app

SMARTIESDIR = ../../source/Communicators/
CPPFLAGS += -std=c++11 -I$(SMARTIESDIR) -DSMARTIES_APP
VPATH := $(VPATH):$(SMARTIESDIR)

app: cart-pole.o cart_pole.o
	ar rs libsimulation.a $^
	cp libsimulation.a ../../makefiles/libsimulation.a
	#$(CC) -shared -c cart-pole.cpp -o app.so -I ../../source/Communicators/
	#cp app.so ../../makefiles/test_mpi_cart_pole.so

clean:
	rm -f *.o *.mod libsimulation.a

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@
%.o: %.f90
	mpif90 -O2 -c $< -o $@
