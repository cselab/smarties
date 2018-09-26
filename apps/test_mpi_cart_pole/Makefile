.DEFAULT_GOAL := app

SMARTIESDIR = ../../source/Communicators/
CPPFLAGS += -std=c++11 -I$(SMARTIESDIR) -DSMARTIES_APP
VPATH := $(VPATH):$(SMARTIESDIR)

app: cart-pole.o
	ar rs libsimulation.a $^
	#$(CC) -shared -c cart-pole.cpp -o app.so -I ../../source/Communicators/
	cp libsimulation.a ../../makefiles/libsimulation.a
	#cp app.so ../../makefiles/test_mpi_cart_pole.so

clean:
	rm -f *.o

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@
