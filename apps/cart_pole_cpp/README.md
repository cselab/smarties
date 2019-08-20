# *c++* application
* This workflow is correct for 1 agent per simulation. Note that while sending state you have to specify whether it is an initial state, normal state, terminal state.
* Socket number is to create a secure channel of communication between smarties and your application, might be moved inside `Communicator` in future releases.
* `state_vars` denotes the dimensionality of the state space. Ie. if you state is `x` and `y` of an agent, `state_vars = 2`.
* `control_vars` denotes the dimensionality of the action space. Ie. if you control the `v_x` and `v_y` of the agent, `control_vars = 2`.
* Optional stuff. Commands to be used before sending the first state.
    - Action bounds. Useful if you know the range of meaningful actions.
    ```
    comm.set_action_scales(upper_action_bound, lower_action_bound, bounded);
    ```
    Ie. in the cart pole problem, the actions are in the range -10 to 10. With RL left with default parameters, it will start exploration with actions distributed with mean 0 and std deviation 1. If you submit this command with `vector<double> upper_action_bound{10}` and `lower_action_bound{-10};`, the action space will be rescaled to that range, and the exploration will effectively be with std deviation 10.
    If `bool bounded` is set to true, actions can ONLY be performed in the specified range.
    - Action options. This command works for discrete action spaces.
    ```
    comm.set_action_options(n_options);
    ```
    `n_options` can either be an integer or a vector of integers, with the same size as the dimensionality of the action space. This means that, if the agent can control 2 numbers, and for each number you want to provide two options: `vector<int> n_options{2, 2}`.
    - Observability of state variables. Some state variables might not be observable to the agent, or you might want to pass additional data to smarties for logging, but not include it in the state vector.
    ```
    comm.set_state_observable(b_observable);
    ```
    `b_observable` is a `vector<bool>` with dimensionality `state_vars`.
    - Input to neural networks should be normalized. If you know the range of the state variables, provide it as:
    ```
    comm.set_state_scales(upper_state_bound, lower_state_bound);
    ```
    The two arguments are `vector<double>` with dimensionality `state_vars`. If not provided, range is assumed to be -1 to 1 per variable.
* You can provide in the constructor a number of agents. Ie:
    ```
    Communicator comm(socket, state_vars, control_vars, number_of_agents);
    ```
      Multi agent systems require additional care. First of all, smarties does not support environments with independent agents. This means that, while actions request can happen independently (no fixed ordering or equal number of requests), if an agent sends a terminal state, smarties assumes that the environment is over: all other agents are about to send a terminal state and restart. However, terminal states are special states in reinforcement learning, because `V(term_state) = 0`. If some agents reach a terminal state and the others do not, first send all the terminal states, then use the function `comm.truncateSeq(observation, reward)`. smarties will treat the trajectories of the other agents, those which did not send a terminal state, as interrupted sequences. This means that the value of the last state will not be 0, but rather the expected on policy state value. E.g: in an environment with multiple cars, if 2 cars crash they reach a terminal state, but after the crash the environment will still be restarted for all cars.

* You can provide application specific settings by letting smarties find the file `appSettings.sh` in the run directory. See the two cartpoles example.
