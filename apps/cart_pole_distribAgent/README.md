# apps

Specific example of case where each agent requires a different policy. These are simply two cart poles, except that agent 0 applies -action while agent 1 applies action. Smarties' policies are not told which agent is requesting an action, therefore if only one policy is used to solve this problem either only one agent will learn, or none of them will learn.  

If the runtime setting option `bSharedPol` is set to 0, two learning algorithms will be created and both agents will be able to solve the task. This is also an example of an environment needing specific smarties settings and therefore writing the `appSettings.sh` file.

In theory, it should be possible to have multiple agents sharing a single environment either to compete or collaborate. The main limitation is that for now the state and action description must be the same for both.
