# Communicators

Communicators are objects that define the exchange of data between the RL algorithms of smarties and the simulation of the environment.

#### Python Communicators
 
Initialization functions:
  - `__init__(self, state_components, action_components, number_of_agents=1, discrete_actions=False)` is the constructor which asks for the dimensionality of the state and action space. Note the case of discrete action spaces where the agent can perform one out of N options. Since the agent can only select one option per turn the dimensionality is 1. If the agent were to be able to select two options per turn (e.g. turn L/R AND accel/decel) the dimensionality is 2 and so on.
  - `set_action_scales(self, upper, lower, bounded=False)` takes two vectors of size `action_components` that define the scales of the action space. If the optional argument is set to true, these scales are the limiting bounds of the action space box.
  - `set_state_scales(self, upper, lower)` takes the scales of the state space. May help the NN learn.
  - `set_state_observable(self, observable)` allows hiding some of the components of the state vector from the agent. Possible uses: 1) test RNN by making the problem partially observable. 2) Giving additional information and observables to smarties, which will be stored to file, which should not be included in the state. This can be used for post processing, batch RL, or IRL.

Training functions:
  - `sendInitState(self, observation, agent_id=0)` sends the initial state in an episode. Since it happens before any action is performed, the reward is by definition 0.
  - `sendTermState(self, observation, reward, agent_id=0)` sends a terminal state of an episode. This means that the task is concludes due to failure or success.
  - `truncateSeq(self, observation, reward, agent_id=0)` sends the last state in a truncated episode. Truncated episodes may exist because of time limits in the simulation of the environment. For example, we might train a RL agent to walk and have the simulations last at most T steps. If we were to lift this time limit from the simulation however we would like the controller to continue walking unaffected by the change. Conversely, if the agent must conclude a task before T steps otherwise it fails then that's a termination condition. Otherwise, truncated episodes arise from multi-agent environments. For example when one agent fails/succeeds causing the episode to conclude, but this may not mean that the other agents have succeeded or failed.  
  - `sendState(self, observation, reward, terminal=False, initial=False, truncated=False, agent_id=0)` default behavior is to send an `intermediate` state in an episode.
  - `recvAction(self, agent_id=0)` must be used after sending a state.

For convenience there are specialized Communicators: `Communicator_gym` creates and handles an instance of the OpenAI gym. `Communicator_atari` uses OpenAI gym for Atari games and applies the DQN preprocessing. `Communicator_dmc` is for the DeepMind Control Suite.


#### C++ Communicators
Defines the same functions as above but in C++.

#### Internal Communicator
Created by smarties and used internally. 