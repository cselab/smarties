Automating Turbulence Modeling by Multi-Agent Reinforcement Learning
********************************************************************

This folder contains all training and post-processing scripts, as well as settings file, used to obtain the results in the paper 
`Automating Turbulence Modeling by Multi-Agent Reinforcement Learning <https://arxiv.org/pdf/2005.09023.pdf>`_. 
Because ``smarties`` is a library, the main executable that simulates the flow is produced by 
`CubismUP 3D <https://github.com/cselab/CubismUP_3D>`_.
We refer to that page for instructions regarding install and dependencies. 
The dependencies of ``CubismUP 3D`` are a superset of those of ``smarties``.

All the scripts in this folder assume that:  

- The directories of ``smarties`` and ``CubismUP 3D`` are placed at the same path (e.g. ``${HOME}/``).  
- ``smarties`` is installed and ``CubismUP 3D`` can be compiled without issues.

Core task description
=====================
The ``CubismUP 3D`` file ``source/main_RL_HIT.cpp`` produces the main executable.
It describes the environment loop and parses all hyper-parameters.
This file interacts with 3 objects:

- The class ``smarties::Communicator`` receives a description of the RL problem and handles the state-action loop for all the agents.   
- The class ``cubismup3d::Simulation`` comprises the solver and defines the operations performed on each (simulation) time-step.   
- The class ``cubismup3d::SGS_RL`` describes the operations performed to update the Smagorinsky coefficients. This class describes both the interpolation of the actions onto the grid and the calculation of the state components.


In this folder, the file ``setup.sh`` is read by ``smarties.py`` and prepares all hyper-parameters, and simulation description.

Training script
===============
Training can be easily started, with default hyper-parameters, as:

.. code:: shell

    smarties.py CUP3D_LES_HIT -r training_directory_name

In order to reproduce the number of gradient steps of the paper on a personal computer, the script may run for days, 
with large uncertainty due to the specific processor and software stack. We relied on the computational resources provided by
the Swiss National Supercomputing Centre (CSCS) (on the Piz Daint supercomputer).
We provide a set of trained policy parameters and restart folder.
The helper file ``launch_all_les_blocksizes.sh`` was used to evaluate  multiple hyper-parameter choices.

By default, ``smarties.py`` will place all run directories in ``${SMARTIES_ROOT}/runs``, but can be changed with
the argument ``--runprefix``.

When training, the terminal output will be that of ``smarties.py``, which tracks training progress, not of ``CubismUP 3D``.
The terminal output of the simulations is redirected to, for example, ``training_directory_name/simulation_000_00000/output`` and 
all the simulation snapshots and post-processing to, for example, ``training_directory_name/simulation_000_00000/run_00000000/``.
During training, no post-processing (e.g. energy spectra, dissipation, other integral statistics) are stored to file.

Running the trained model
==========================
Once trained, the policy can be used to perform any simulation. This can be done for example as:

.. code:: shell

    smarties.py CUP3D_LES_HIT -r evaluation_directory_name --restart training_directory_name --nEvalEpisodes 1

This process should take few minutes. Again, the terminal output will be that of ``smarties.py``,
which, if everything works correctly, will not be very informative.
To see the terminal output of the simulation itself prepend ``--printAppStdout`` to the run command.

Because we specificed that we evaluate the policy for 1 episode (or N), training is disabled and the policy is fixed.
However, the ``CubismUP_3D`` side will run identically as for training, which means that it will simulate a random Reynolds number.
Using the script

.. code:: shell

    python3 eval_all_train.py training_directory_name

will evaluate the policy saved in ``training_directory_name`` at Reynolds in log-intervals from 60 to 205, each in a separate directory.
Each evaluation directory will be named according to the Reynolds like: ``training_directory_name_RE%03d``.

Evaluating the trained policy
==============================
From the list of directories, the energy spectra can be plotted as

.. code:: shell

   python3 plot_eval_all_les.py training_directory_name --runspath /rel/path/to/runs/ --res 65, 76, 88, 103, 120, 140, 163

Here we need to write the relative path to where ``smarties.py`` has created the evaluation runs.
By default, all the directories were placed in ``${SMARTIES_ROOT}/runs``.
