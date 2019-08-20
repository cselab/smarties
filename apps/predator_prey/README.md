# apps

Specific example of case where each agent requires a different policy. These are simply two cart poles, except that agent 0 applies -action while agent 1 applies action. Smarties' policies are not told which agent is requesting an action, therefore if only one policy is used to solve this problem either only one agent will learn, or none of them will learn.  

If the runtime setting option `bSharedPol` is set to 0, two learning algorithms will be created and both agents will be able to solve the task. This is also an example of an environment needing specific smarties settings and therefore writing the `appSettings.sh` file.

In theory, it should be possible to have multiple agents sharing a single environment either to compete or collaborate. The main limitation is that for now the state and action description must be the same for both.

# termination vs truncation

* If the simulation has reached the maximum number of iterations allowed (e.g., none of the predators manage to capture a prey), the simulation has reached a truncation state. All agents must call `comm.truncateSeq(state, reward, #);`, and break the simulation loop to reach `comm.sendInitState(state);` again.

* If the simulation is over, i.e., it has reached a termination state, we must provide the terminal reward to all the agents involved. This can involve some agents receiving a large positive/negative reward, and some receiving no additional reward. This happens because some agents will be in a terminal state because they failed or succeeded, whereas some agents might be still alive and are neither winners nor losers (e.g. multiple prey where only one died, the others did not fail). When the simulation is over, the agents that are in a terminal state call `comm.sendTermState(state, reward, #);`. This reward can be special (bonus, penalty), or can be the normal reward. For the agents who are neither winners nor loosers, `comm.truncateSeq(state, reward, #);` should be called. The simulation loop is then broken to reach `comm.sendInitState(state);` again.
