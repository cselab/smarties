# launch

The main files that will be maintained are `launch.sh`, `launch_(gym/dmcs/atari).sh`, `launchDaint.sh`, `run.sh`.

`launch.sh` is the main launch script. You must provide:
* the name of the folder to run in, which will be placed in `../runs/`.
* the path or name of the folder in the `apps` folder containing the files defining your application.
* the path to the settings file.  

Then it accepts the following optional arguments:  
* (optional, default 1) the number of worker ranks per learner rank. If the environment application does not require multiple ranks itself (ie. does not require MPI), it means number of environment instances per learner.  
    - Must be at least 1. If the environment requires multiple ranks itself (ie. MPI app) then the number of workers must be a multiple of the number of ranks required by each instance of the application (smarties reads this from the settings file).  
    - More than one worker rank per learner might be useful if the simulations are particularly slow.  
* (optional, default the number of physical cores read from lscpu) the number of omp-threads (>=1) over which the grad update compute is distributed (therefore your mpi distribution must support thread safety). This does not control the number of threads (if any) used by app.
* (optional, default 1) the number of learner ranks. Unless the network is very large this should not need to change.
* (optional, default 1) the number of nodes to use. This setting affects the `ppn` option given to `mpirun`.

* `launch_gym.sh` behaves much the same way, but instead of providing a path to an application provide the name of the OpenAI Gym environment (e.g. `CartPole-v1`)

These two scripts set up the launch environment and directory, and then call `run.sh`.

* `launchDaint.sh` .. it works on CSCS piz Daint. Main changes are that run folder is in `/scratch/snx3000/${MYNAME}/smarties/`, the number of threads is hardcoded to 12, and `run.sh` is not used.

* An example of running a `C++` based app is `./launch.sh RUNDIR test_cpp_cart_pole settings/settings_RACER.sh` . To see an example of how to set up a `C++` app see the folder `../apps/`. The setting file `settings/settings_RACER.sh` details the baseline solver of `smarties`.

* An example of launching an OpenAI gym based app is `./launch_openai.sh RUNDIR Walker2d-v2 settings/settings_RACER.sh` .

* `settings/settings_VRACER.sh` details the simplified V-Racer architecture. Can speed up learning. Easier to explain.

* The best strategy to speed up learning for  _easy problems_ is to change `--gamma 0.99`, `--maxTotObsNum 262144`

# outputs

* Running the script will produce the following outputs on screen (also backed up into the files `agent_%02d_stats.txt`). According to applicability, these are either statistics computed over the past 1000 steps or are the most recent values:
    - `ID` Learner identifier. If a single environment contains multiple agents, and if each agent requires a different policy (--bSharedPol 0), then we distinguish outputs pertinent to each agent with this ID integer.
    - `#/1e3` Counter of gradient steps divided by 1000
    - `avgR` Average _SCALED_ cumulative reward among stored episodes (shown scaled for brevity and generality).
    - `stdR`  Std dev of the distribution of instantaneous rewards. The unscaled average cumulative rewards is avgR x stdR.
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

# misc

* To evaluate the learned behaviors of a concluded training run we have to restore the internal state of smarties. Since the `agent_%02d_*` files contain all the information to recover the correct state/reward rescaling and the network weights we call them 'policy files'. Once read, they allow smarties to recover the same policy as during training. Steps:
    - (1) Make sure `--bTrain 0`
    - (2) (optional) `--explNoise 0` if the agents should deterministically perform the most probable discrete action or the mean of the Gaussian policy.
    - (3) For safety, copy over all the `agent_%02d_*` files onto a new folder in order to not overwrite any file of the training directory and select this new folder as the run directory (ie. arg $1 of launch.sh ). 
    - (3) Otherwise, the setting `--restart /path/to/dir/` (which defaults to "." if `bTrain==0`) can be used to specify the path to the `agent_%02d_*` files without having to manually copy them over into a new folder.
    - (4) Run with at least one mpi-rank for the master plus the number of mpi-ranks for one instance of the application (usually 1).
    - (5) To run a finite number of times, the option `--totNumSteps` is recycled if `bTrain==0` to be the number of sequences that are observed before terminating (instead of the maximum number of gradient steps done for the training if `bTrain==1`)
    - (6) Make sure the policy is read correctly (eg. if code was compiled with different features or run with different algorithms, network might have different shape), by comparing the `restarted_policy...` files and the original `agent_%02d_*` files. This can be performed with the `diff` command (ie. `diff /path/eval/run/restarted_net_weights.raw /path/train/run/agent_00_net_weights.raw`).

* For a description of the settings read `source/Settings.h`. The file follows 	an uniform pattern:
	```
	#define CHARARG_argName 'a'
	#define COMMENT_argName "Argument description"
	#define TYPEVAL_argName int
	#define TYPENUM_argName INT
	#define DEFAULT_argName 0
	int argName = DEFAULT_argName;
	```

    - The first line (`CHARARG_`) defines the `char` associated with the argument, to run `./rl -a val`
    - The second line (`COMMENT_`) contains a brief description of the argument.
    - The third line (`TYPEVAL_`) contains the variable type: must be either `int`, `string`, `Real`, or `char`.
    - The fourth line (`TYPENUM_`) specifies the enumerator in the argument parser associated with the variable type: `INT`, `STRING`, `REAL`.
    - The fifth line (`DEFAULT_`) specifies the default value for the argument.
    - The sixth and last line assigns in the constructor of `Settings` the default value to the variable.
    - Later in the text `ArgumentParser` is called to read from the arguments. The string that defines each argument is `argName`. Therefore run as `./rl --argName val`.

* The `_client.sh` scripts will temporarily no longer be supported since their usefulness was outweighed by the confusion of the users. Also, by design it was not possible to make it work on Daint out of the box without the user writing its own `launchDaint.sh` for the application.
