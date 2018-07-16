# Internal application

Example of environment which linked at compile-time to the main smarties code.
This is the hackiest setup, but it may be necessary for applications that require MPI to run. The main method of the environment now reads:  
`int app_main(Communicator*const comm, MPI_Comm mpicom, int argc, char**argv)`
* `comm` can be used like in the other setups to set the problem up and being the sendState/recvAction loop.  
* `mpicom` is the MPI communicator that allows groups of workers to work on the same simulation. MPI codes should use that communicator for all inter-rank communication of data. Using `MPI_COMM_WORLD` will lead to undefined behavior because WORLD is shared by the master rank or other simulations.  
* `argc` and `argv` are read from a settings file. More on this later.
Moreover, the top of the file should link the relative path for the header:  
`#include "../../source/Communicators/Communicator.h"`  


There are two tools to make defining MPI-based environments easier.  
* The setting `appSettings` (which can be read from smarties settings file) specifies the name of a file that is read before launching `app_main`. If the file is found, all its contents are passed as command line arguments to `app_main`.  
* The setting `setupFolder` specifies the path to a folder whose contents are copied inside the simulation's run folder. This can be used to copy over configuration files or other files that the MPI based application needs to read on each start up.  


The `setup.sh` file is the hackiest part of this setup. It should call smarties' Makefile and provide the path to the main object file created by compiling this application. I.e. by default here we do:  
`make -C ../makefiles/ app=../apps/test_mpi_cart_pole/cart-pole -j config=prod`  
The path to smarties' Makefile is `../makefiles` because this `setup.sh` is source-d from the `launch` directory. Here we just provide the path `cart-pole` and smarties' Makefile will compile `%.cpp` into `%.o`. If compilation is more complex (e.g. it involves multiple object files), `setup.sh` can call the application's Makefile, copy static library file into `../makefiles`, and smarties' Makefile can be modified to link to that object file. This is how we link smarties to CubismUP_3D.
