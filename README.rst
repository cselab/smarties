smarties
**********

**smarties is a Reinforcement Learning (RL) software designed with the following
objectives**

- high-performance C++ implementations of [Remember and Forget for Experience Replay](https://arxiv.org/abs/1807.05827) and other deep RL learning algorithms including V-RACER, CMA, PPO, DQN, DPG, ACER, and NAF.

- the environment application determines at runtime the properties of the control problem to be solved. For example, the number of the agents in the environment, whether they are solving the same problem (and therefore they should all contribute to learning a common policy) or they are solving different problems (e.g. competing or collaborating). The properties of each agent's state and action spaces. Whether one or more agents are dealing with a partially-observable problem, which causes the learners to automatically use recurrent networks for function approximation. Whether the observation has to be preprocessed by convolutional layers and the properties thereof.

- the environment application is in control of the learning progress. More specifically, smarties supports applications whose API design is similar to that of OpenAI gym, where the environment is a self-contained function to be called to receive a new observation and advance the simulation in time. However, smarties also supports a more realistic API where the environment simulation dominates the structure of the application code. In this setting, it is the RL code that is called whenever new observations are available and require the agent to pick an action.

- the environment application determines the  computational resources available to train and run simulations, with support for distributed (MPI) codebases.

- minimally intrusive plug-in API that can be inserted into C++, python and Fortran simulation software.  

To cite this repository, reference the paper::

    @inproceedings{novati2019a,
        title={Remember and Forget for Experience Replay},
        author={Novati, Guido and Koumoutsakos, Petros},
        booktitle={Proceedings of the 36\textsuperscript{th} International Conference on Machine Learning},
        year={2019}
    }

.. contents:: **Contents of this document**
   :depth: 3

Install
======

Regardless of OS, smarties relies on pybind11 for the Python interface. Therefore, to enable compilation of smarties with Python environment simulation codebases:

.. code:: shell

    pip3 install pybind11 --user

Moreover, after the compilation steps for either Linux or Mac OS, add the path to the smarties library directory to your dynamic loader:

.. code:: shell

    echo 'export SMARTIES_ROOT=/path/to/smarties/folder/' >> ~/.bash_profile
    echo 'export LD_LIBRARY_PATH=${SMARTIES_ROOT}/lib:${LD_LIBRARY_PATH}' >> ~/.bash_profile

The environment variable 'SMARTIES_ROOT' is used to compile most of the applications in the 'apps' folder.

Linux
------

Smarties requires gcc version 6.1 or greater, a thread-safe (at least `MPI_THREAD_SERIALIZED`) implementation of MPI, and a serial BLAS implementation with CBLAS interface. Furthermore, in order to test on the benchmark problems, OpenAI gym or the DeepMind Control Suite with python>=3.5. MPI and OpenBLAS can be installed by running the ``install_dependencies.sh`` script.

.. code:: shell

    git clone https://github.com/cselab/smarties.git
    cd smarties
    mkdir -p build
    cd build
    cmake ..
    make -j

Mac OS
------
Installation on Mac OS is a bit more laborious due to to the LLVM compiler provided by Apple not supporting OpenMP threads. First, install the required dependencies as:

.. code:: shell

    brew install llvm libomp open-mpi openblas

Now, we have to switch from Apple's LLVM compiler to the most recent LLVM compiler as default for the user's shell:

.. code:: shell

    echo "alias cc='/usr/local/opt/llvm/bin/clang'" >> ~/.bash_profile
    echo "alias gcc='/usr/local/opt/llvm/bin/clang'" >> ~/.bash_profile
    echo "alias g++='/usr/local/opt/llvm/bin/clang++'" >> ~/.bash_profile
    echo "alias c++='/usr/local/opt/llvm/bin/clang++'" >> ~/.bash_profile
    echo "export PATH=/usr/local/opt/llvm/bin:\${PATH}" >> ~/.bash_profile

Then we are ready to get and install smarties:

.. code:: shell

    git clone https://github.com/cselab/smarties.git
    cd smarties/makefiles
    make -j


User code samples
=================

C++
-----
The basic structure of a C++ based application for smarties is structured as:

.. code:: shell

    #include "smarties.h"
    
    inline void app_main(smarties::Communicator*const comm, int argc, char**argv)
    {
      comm->set_state_action_dims(state_dimensionality, action_dimensionality);
      Environment env;
    
      while(true) { //train loop
        env.reset(comm->getPRNG()); // prng with different seed on each process
        comm->sendInitState(env.getState()); //send initial state
    
        while (true) { //simulation loop
          std::vector<double> action = comm->recvAction();
          bool isTerminal = env.advance(action); //advance the simulation:
    
          if(isTerminal) { //tell smarties that this is a terminal state
            comm->sendTermState(env.getState(), env.getReward());
            break;
          } else  # normal state
            comm->sendState(env.getState(), env.getReward());
        }
      }
    }
    
    int main(int argc, char**argv)
    {
      smarties::Engine e(argc, argv);
      if( e.parse() ) return 1;
      e.run( app_main );
      return 0;
    }

For compilation, the following flags should be set in order for the compiler to find smarties:

.. code:: shell

    LDFLAGS="-L${SMARTIES_ROOT}/lib -lsmarties"
    CPPFLAGS="-I${SMARTIES_ROOT}/include"


Python  
-----
smarties uses pybind11 for seamless compatibility with python. The structure of the environment application is almost the same as the C++ version:

.. code:: shell

    import smarties as rl
    
    def app_main(comm):
      comm.set_state_action_dims(state_dimensionality, action_dimensionality)
      env = Environment()
    
      while 1: #train loop
        env.reset() # (slightly) random initial conditions are best
        comm.sendInitState(env.getState())
    
        while 1: #simulation loop
          action = comm.recvAction()
          isTerminal = env.advance(action)
    
          if terminated:  # tell smarties that this is a terminal state
            comm.sendTermState(env.getState(), env.getReward())
            break
          else: # normal state
            comm.sendState(env.getState(), env.getReward())
    
    if __name__ == '__main__':
      e = rl.Engine(sys.argv)
      if( e.parse() ): exit()
      e.run( app_main )


Code examples
--------------
The ``apps`` folder contains a number of examples showing the various use-cases of smarties. Each folder contains the files required to define and run a different application. While it is generally possible to run each case as ``./exec`` or ``./exec.py``, smarties will create a number of log files, simulation folders and restart files. Therefore it is recommended to manually create a run directory or use the launch scripts contained in the ``launch`` directory.

The applications that are already included are:

- ``apps/cart_pole_cpp``: simple C++ example of a cart-pole balancing problem  

- ``apps/cart_pole_py``: simple python example of a cart-pole balancing problem  

- ``apps/cart_pole_f90``: simple fortran example of a cart-pole balancing problem  

- ``apps/cart_pole_many``: example of two cart-poles that define different decision processes: one performs the opposite of the action sent by smarties and the other hides some of the state variables from the learner (partially observable) and tehrefore requires recurrent networks.  

- ``apps/cart_pole_distribEnv``: example of a distributed environment which requires MPI. The application requests M ranks to run each simulation. If the executable is ran as ``mpirun -n N exec``, (N-1)/M teams of processes will be created, each with its own MPI communicator. Each simulation process contains one or more agents.  

- ``apps/cart_pole_distribAgent``: example of a problem where the agent themselves are distributed. Meaning that the agents exist across the team of processes that run a simulation and get the same action to perform. For example flow actuation problems where there is only one control variable (eg. some inflow parameter), but the entire simulation requires multiple CPUs to run.  

- ``apps/predator_prey``: example of agents competing.  

- ``apps/glider``: example of an ODE-based control problem that requires precise controls, used for the paper [Deep-Reinforcement-Learning for Gliding and Perching Bodies](https://arxiv.org/abs/1807.03671)  

- ``apps/func_maximization/``: example of function fitting and maximization, most naturally approached with CMA.  

- ``apps/OpenAI_gym``: code to run most gym application, including the MuJoCo based robotic benchmarks shown in [Remember and Forget for Experience Replay](https://arxiv.org/abs/1807.05827)  

- ``apps/OpenAI_gym_atari``: code to run the Atari games, which automatically creates the required convolutional pre-processing  

- ``apps/Deepmind_control``: code to run the Deepmind Control Suite control problems

- ``apps/CUP2D_2fish``: and similarly named applications require `CubismUP 2D <https://github.com/novatig/CubismUP_2D>`_.

Examples of solved problems
---------------------------

.. raw:: html

    <a href="https://www.youtube.com/watch?v=H9xL9nNQJnc"><img src="https://img.youtube.com/vi/H9xL9nNQJnc/0.jpg" alt="V-RACER trained on OpenAI gym's Humanoid-v2"></a>

.. raw:: html

    <a href="https://www.youtube.com/watch?v=5mK9HoCDIYQ"><img src="https://img.youtube.com/vi/5mK9HoCDIYQ/0.jpg" alt="Smart ellipse behind a D-section cylinder. Trained with V-RACER."></a>

.. raw:: html

    <a href="https://www.youtube.com/watch?v=GiS9mxQ4m0I"><img src="https://img.youtube.com/vi/GiS9mxQ4m0I/0.jpg" alt="Fish behind a  D-section cylinder"></a>

.. raw:: html

    <a href="https://www.youtube.com/watch?v=NEOhS0kPrSk"><img src="https://img.youtube.com/vi/NEOhS0kPrSk/0.jpg" alt="Smart swimmer following an erratic leader to minimize swimming effort."></a>

.. raw:: html

    <a href="https://www.youtube.com/watch?v=8pKhMgPm5p0"><img src="https://img.youtube.com/vi/8pKhMgPm5p0/0.jpg" alt="3D fish schooling"></a>

Launching
=========

It is possible to run smarties without the tools defined in this folder.
However, the script ``launch.sh`` provides some functionality that helps running
smarties on multiple processes. For example having multiple processes running
the environment (to parallelize data-collection) or multiple processes hosting
the RL algorithms (to parallelize gradient descent).  

When using `launch.sh` you may provide:

* the name of the folder to run in, which by default will be placed in `runs/`.

* the path or name of the folder in the `apps` folder containing the files defining your application.

* (optional) the path to the settings file. The default setting file, specifying the RL solver and hyper-parameters, is set to `settings/VRACER.json`.

* (optional) the number of threads that should be used by the learning algorithm on each process to update the networks.

* (optional) the number of computational nodes available to run the training.

* (optional) the number of dedicated MPI processes dedicated to update the networks. If the network, the batchsize, or the CMA population size are large it might be beneficial to add more master processes. The memory buffer and the batch size will be spread among all learners. Once an experience is stored by a learning process it will never be moved again.

* (optional) the number of worker MPI processes that run the simulation. If the environment application does not require multiple ranks itself (ie. does not require MPI), it means number of separate environment instances. Many off-policy algorithms require a certain number of environment time steps per gradient steps, these are uniformly distributed among worker processes (ie. worker ranks may alternate in advancing their simulation). Must be at least 1. If the environment requires multiple ranks itself (ie. MPI app) then the number of workers must be a multiple of the number of ranks required by each instance of the application.

For example:

.. code:: shell

    ./launch.sh testRun cart_pole_cpp RACER.json 4 1 1 1

Which will setup the folder runs/testRun and run the ``cart_pole_cpp`` example on
one process (``cart_pole_cpp`` does not require dedicated MPI processes), with
one running simulation of the cart-pole, and 4 threads doing the gradient descent updates. The script will call:

.. code:: shell

    mpirun -n 1 --map-by ppr:1:node ./exec --nWorkers 1 --nMasters 1 --nThreads 4

Additional remarks:

* An example of launching an OpenAI gym mujoco-based app is `./launch_gym.sh RUNDIR Walker2d-v2`. The second argument, instead of providing a path to an application, is the name of the OpenAI Gym environment (e.g. `CartPole-v1`)

* An example of launching an OpenAI gym Atari-based app is `./launch_atari.sh RUNDIR Pong` (the version specifier `NoFrameskip-v4` will be added internally). Note that we apply the same frame preprocessing as in the OpenAI `baselines` repository and the base CNN architecture is the same as in the DQN paper. The network layers specified in the `settings` file (ie. fully connected, GRU, LSTM) will be added on top of those convolutional layers.


* (optional, default 1) `nMasters`: the number of learner ranks. 
* (optional, default 1) `nWorkers`: the total number of environment processes. 

These two scripts set up the launch environment and directory, and then call `run.sh`.

Learning outputs
=======

* Running the script will produce the following outputs on screen (also backed up into the files `agent_%02d_stats.txt`). According to applicability, these are either statistics computed over the past 1000 steps or are the most recent values:
    - `ID` Learner identifier. If a single environment contains multiple agents, and if each agent requires a different policy (--bSharedPol 0), then we distinguish outputs pertinent to each agent with this ID integer.
    - `#/1e3` Counter of gradient steps divided by 1000
    - `avgR` Average **cumulative** reward among stored episodes.
    - `stdr`  Std dev of the distribution of **instantaneous** rewards. The unscaled average cumulative rewards is `avgR` x `stdr`.
    - `DKL` Average Kullback Leibler of samples in the buffer w.r.t. current policy.
    - `nEp |  nObs | totEp | totObs | oldEp | nFarP` Number of episodes and observations in the Replay Memory. Total ep/obs since beginning of training passing through the buffer. Time stamp of the oldest episode (more precisely, of the last observation of the episode) that is currently in the buffer. Number of far policy samples in the buffer.
    - `RMSE | avgQ | stdQ | minQ | maxQ` RMSE of Q (or V) approximator, its average value, standard deviation, min and max.
    - (if algorithm employs parameterized policy) `polG | penG | proj` Average norm of the policy gradient and that of the penalization gradient (if applicable). Third is the average projection of the policy gradient over the penalty one. I.e. the average value of `proj = polG \cdot penG / sqrt(penG \cdot penG) `. `proj` should generally be negative: current policy should be moved away from past behavior in the direction of pol grad.
    - (extra outputs depending on algorithms) In RACER/DPG: `beta` is the weight between penalty and policy gradients. `avgW` is the average value of the off policy importance weight `pi/mu`. `dAdv` is the average change of the value of the Retrace estimator for a state-action pair between two consecutive times the pair was sampled for learning. In PPO: `beta` is the coefficient of the penalty gradient. `DKL` is the average Kullback Leibler of the 'proximally' on-policy samples used to compute updates. `avgW` is the average value of `pi/mu`. `DKLt` is the target value of Kullback Leibler if algorithm is trying to learn a value for it.
    - `net` and/or `policy` and/or `critic` and/or `input` and/or other: L2 norm of the weights of the corresponding network approximator.

* The file `agent_%02d_rank%02d_cumulative_rewards.dat` contains the all-important cumulative rewards. It is stored as text-columns specifying: gradient count, time step count, agent id, episode length (in time steps), sum of rewards over the episode. The first two values are recorded when the last observation of the episode has been recorded. Can be plotted with the script `pytools/plot_rew.py`.

* The files `${network_name}_grads.raw` record the statistics (mean, standard deviation) of the gradients received by each network output. Can be plotted with `pytools/plot_grads.py`.

* If the option `--samplesFile 1` is set, a complete log of all state/action/rewards/policies will be recorded in binary files named `obs_rank%02d_agent%03d.raw`. This is read by the script `pytools/plot_obs.py`. Refer also to that script (or to `source/Agent.h`) for details on the structure of these files.

* The files named `agent_%02d_${network_name}_${SPEC}_${timestamp}` contain back-ups of network weights (`weights`), Adam's moments estimates (`1stMom` and `2ndMom`) and target weights (`tgt_weights`) at regularly spaced time stamps. Some insight into the shape of the weight vector can be obtained by plotting with the script `pytools/plot_weights.py`. The files ending in `scaling.raw` contain the values used to rescale the states and rewards. Specifically, one after the other, 3 arrays of size `d_S` of the state-values means, 1/stdev, and stdev, followed by one value corresponding to 1/stdev of the rewards.

* Various files ending in `.log`. These record the state of smarties on startup. They include: `gitdiff.log` records the changes wrt the last commit, `gitlog.log` records the last commits, `mathtest.log` tests for correctness of policy/advantage gradients, `out.log` is a copy of the screen output, `problem_size.log` records state/action sizes used by other scripts, `settings.log` records the runtime options as read by smarties, `environment.log` records the environment variables at startup.

Misc
====

* To evaluate the learned behaviors of a concluded training run we have to restore the internal state of smarties. Since the `agent_%02d_*` files contain all the information to recover the correct state/reward rescaling and the network weights we call them 'policy files'. Once read, they allow smarties to recover the same policy as during training. Steps:
    - (1) Make sure `--bTrain 0`
    - (2) (optional) `--explNoise 0` if the agents should deterministically perform the most probable discrete action or the mean of the Gaussian policy.
    - (3) For safety, copy over all the `agent_%02d_*` files onto a new folder in order to not overwrite any file of the training directory and select this new folder as the run directory (ie. arg $1 of launch.sh ).
    - (3) Otherwise, the setting `--restart /path/to/dir/` (which defaults to "." if `bTrain==0`) can be used to specify the path to the `agent_%02d_*` files without having to manually copy them over into a new folder.
    - (4) Run with at least one mpi-rank for the master plus the number of mpi-ranks for one instance of the application (usually 1).
    - (5) To run a finite number of times, the option `--totNumSteps` is recycled if `bTrain==0` to be the number of sequences that are observed before terminating (instead of the maximum number of time steps done for the training if `bTrain==1`)
    - (6) Make sure the policy is read correctly (eg. if code was compiled with different features or run with different algorithms, network might have different shape), by comparing the `restarted_policy...` files and the original `agent_%02d_*` files. This can be performed with the `diff` command (ie. `diff /path/eval/run/restarted_net_weights.raw /path/train/run/agent_00_net_weights.raw`).
* Te restart training prepare a folder with the latest scaling (`agent_*_scaling.raw`), weight (`agent_00_net_weights.raw`), target net's weights (`agent_00_net_tgt_weights.raw`), and Adam's momenta (`agent_00_net_*Mom.raw`) files. Moreover, move the last stored state of the learners (`agent_*_rank_*_LASTTIMESTEP_learner.raw`) into the new folder removing the time stamp (`agent_00_rank_000_learner.raw`). At this point training can continue as if never interrupted from the last saved step. Make sure you use the same settings file.
* It is possible to begin training anew but use the trained weights of a previous run as a first guess. In this case I found it best not to carry over Adam's momenta files and recover only the weight themselves.

