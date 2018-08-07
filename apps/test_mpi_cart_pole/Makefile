CC ?= mpic++
.DEFAULT_GOAL := app

SMARTIESDIR = ../../source/Communicators/
CPPFLAGS += -I$(SMARTIESDIR)
VPATH := $(VPATH):$(SMARTIESDIR)

app: cart-pole.o Communicator.o
	ar rs libsimulation.a $^
	#$(CC) -shared -c cart-pole.cpp -o app.so -I ../../source/Communicators/
	cp libsimulation.a ../../makefiles/libsimulation.a
	#cp app.so ../../makefiles/test_mpi_cart_pole.so

clean:
	rm -f app.so

%.o: %.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@
