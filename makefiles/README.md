# Dependencies
* **Euler** :
  ```
  module load new modules gcc/6.3.0 mvapich2/2.2 binutils/2.25 hwloc/1.11.0 openblas/0.2.13_seq
  ```
* **Falcon** Have in the bashrc:
	```
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
* **MacOs** Install `mpich` with `gcc`. Assuming gcc version 7:  
    In `~/.profile`:
    ```
    export HOMEBREW_CC=gcc-7
    export HOMEBREW_CXX=g++-7
    ```
    Then `brew install mpich --build-from-source` and `brew install openblas` (without openmp).
