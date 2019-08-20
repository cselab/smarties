# Dependencies
* **Euler** :
  ```
  module load new modules gcc/6.3.0 mvapich2/2.2 binutils/2.25 hwloc/1.11.0 openblas/0.2.13_seq
  ```
* **Falcon** Have in the bashrc:
	```
	module load gnu
	module load openmpi # or module load mpich
	export LD_LIBRARY_PATH=/home/novatig/mpich-3.2/gcc7.1_install/lib/:/usr/local/gcc-7.1/lib64/:$LD_LIBRARY_PATH
	export PATH=/usr/local/gcc-7.1/bin/:/home/novatig/mpich-3.2/gcc7.1_install/bin/::$PATH
	```
* **Panda** Have in the bashrc:
	```
	export PATH=/opt/mpich/bin/:$PATH
	export LD_LIBRARY_PATH=/opt/mpich/lib/:$LD_LIBRARY_PATH
	```
* **Daint** Openai's gym should be installed and activated with virtualenv.
	```
	module swap PrgEnv-cray PrgEnv-gnu
	module load daint-gpu python_virtualenv/15.0.3
	```
* **MacOs** Install `gcc`. Then assuming gcc version 8:  
    In `~/.profile`:
    ```
    export HOMEBREW_CC=gcc-8
    export HOMEBREW_CXX=g++-8
    ```
    Then `source ~/.profile`
    Then it used to be simpler (`CXX=g++-8  CC=gcc-8  FC=gfortran-8 brew install openmpi --build-from-source --cc=gcc-8`).  
    Now many users report issues with installing OpenMPI through Brew. The following code should circumvent the issue:  
    `brew install openmpi --build-from-source --cc=gcc-8  --interactive`  
    In the interactive session type:  
    `export CXX=g++-8; export  CC=gcc-8; export  FC=gfortran-8;`  
    `./configure --prefix=/usr/local/Cellar/open-mpi/4.0.0 --disable-silent-rules --enable-ipv6 --with-libevent=/usr/local/opt/libevent --with-sge --with-gnu-ld`  
    Then type `make` and `make install`, and `exit`.  
    Finally, `brew install openblas` (without openmp).
