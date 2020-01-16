// include entire cpp file to have only one obj file to compile
// (improves compatibility)
#include "Communicator.h"
#include "Engine.h"

#define PYBIND11_HAS_OPTIONAL 0
#define PYBIND11_HAS_EXP_OPTIONAL 0
#define PYBIND11_HAS_VARIANT 0

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace py = pybind11;


PYBIND11_MODULE(smarties, m)
{
  py::class_<smarties::Engine>(m, "Engine")
  
    .def(py::init<std::vector<std::string>> (), py::arg("args"),
         "Constructor for smarties' deployment engine.")

    .def("run", ( void (smarties::Engine::*) (
           const std::function<void(smarties::Communicator*const)> & ) )
         & smarties::Engine::run, py::arg("callback"),
         "Gives control to smarties to train on the task defined by callback.")

    //.def("run", ( void (smarties::Engine::*) (
    //       const std::function<void(smarties::Communicator*const, MPI_Comm)>&) )
    //     & smarties::Engine::run, py::arg("callback"),
    //     "Gives control to smarties to train on the task defined by callback.")

    .def("parse",
         & smarties::Engine::parse,
         "Parses the command line and returns 1 if process should exit.")

    .def("setNthreads",
         & smarties::Engine::setNthreads, py::arg("nThreads"),
         "Sets the number of threads that should be employed by each master "
         "to perform the gradient update steps.")

    .def("setNmasters",
         & smarties::Engine::setNmasters, py::arg("nMasters"),
         "Sets the number of master processes.")

    .def("setNenvironments",
         & smarties::Engine::setNenvironments, py::arg("nEnvironments"),
         "Sets the number of environment simulations.")

    .def("setNworkersPerEnvironment",
         & smarties::Engine::setNworkersPerEnvironment,
         py::arg("workerProcessesPerEnv"),
         "Sets the number of worker processes that should share the MPI_Comm "
         "given back to the environment function for distributed simulations.")

    .def("setRandSeed",
         & smarties::Engine::setRandSeed, py::arg("randSeed"),
         "Overrides the seed for the random number generators.")

    .def("setTotNumTimeSteps",
         & smarties::Engine::setTotNumTimeSteps, py::arg("totNumSteps"),
         "Sets the total number of time steps to perform training on.")

    .def("setSimulationArgumentsFilePath",
         & smarties::Engine::setSimulationArgumentsFilePath,
         py::arg("appSettings"),
         "Allows reading line arguments for the environment simulation from a "
         "file. If set and found, these arguments are added to initial argv.")

    .def("setSimulationSetupFolderPath",
         & smarties::Engine::setSimulationSetupFolderPath,
         py::arg("setupFolder"),
         "Allows having a setup directory whose contents are copied to each "
         "simulation run directory before entering the callback.")

    .def("setRestartFolderPath",
         & smarties::Engine::setRestartFolderPath, py::arg("restart"),
         "Sets the path of the restart files of the RL algorithms.")

    .def("setIsTraining",
         & smarties::Engine::setIsTraining, py::arg("bTrain"),
         "Sets whether to run training or evaluating a policy.")

    .def("setIsLoggingAllData",
         & smarties::Engine::setIsLoggingAllData, py::arg("bLogAllData"),
         "Sets whether to store all state/action/reward data onto files.")

    .def("setAreLearnersOnWorkers",
         & smarties::Engine::setAreLearnersOnWorkers,
         py::arg("bLearnersOnWorkers"),
         "Sets whether worker ranks host a copy of the learning algorithm.");

  py::class_<smarties::Communicator>(m, "Communicator")

    .def("sendInitState",
         & smarties::Communicator::sendInitState,
         py::arg("state"), py::arg("agentID") = 0,
         "Send initial state of a new episode for agent # 'agentID'.")

    .def("sendState",
         & smarties::Communicator::sendState,
         py::arg("state"), py::arg("reward"), py::arg("agentID") = 0,
         "Send normal state and reward for agent # 'agentID'.")

    .def("sendTermState",
         & smarties::Communicator::sendTermState,
         py::arg("state"), py::arg("reward"), py::arg("agentID") = 0,
         "Send terminal state and reward for agent # 'agentID'. "
         "NOTE: V(s_terminal) = 0 because episode cannot continue. "
         "For example, agent succeeded in task or is incapacitated.")

    .def("sendLastState",
         & smarties::Communicator::sendLastState,
         py::arg("state"), py::arg("reward"), py::arg("agentID") = 0,
         "Send last state and reward of the episode for agent # 'agentID'. "
         "NOTE: V(s_last) != 0 because it would be possible to continue the "
         "episode. For example, timeout not caused by the agent's policy.")

    .def("recvAction",
         & smarties::Communicator::recvAction,
         py::arg("agentID") = 0,
         "Get an action for agent # 'agentID' given previously sent state.")

    .def("setNumAgents",
         & smarties::Communicator::setNumAgents,
         py::arg("nAgents"), "Set number of agents in the environment.")

    .def("setStateActionDims",
         & smarties::Communicator::setStateActionDims,
         py::arg("dimState"), py::arg("dimAct"), py::arg("agentID") = 0,
         "Set dimensionality of state and action for agent # 'agentID'.")

    .def("setActionScales",
         ( void (smarties::Communicator::*) (
            const std::vector<double>, const std::vector<double>,
            const bool, const int) )
         & smarties::Communicator::setActionScales,
         py::arg("upper_scale"), py::arg("lower_scale"),
         py::arg("areBounds"), py::arg("agentID") = 0,
         "Set lower and upper scale of the actions for agent # 'agentID'. "
         "Boolean arg specifies if actions are bounded between given values.")

    .def("setActionScales",
         ( void (smarties::Communicator::*) (
            const std::vector<double>, const std::vector<double>,
            const std::vector<bool>, const int) )
         & smarties::Communicator::setActionScales,
         py::arg("upper_scale"), py::arg("lower_scale"),
         py::arg("areBounds"), py::arg("agentID") = 0,
         "Set lower and upper scale of the actions for agent # 'agentID'. "
         "Boolean arg specifies if actions are bounded between gien values.")

    .def("setActionOptions",
         ( void (smarties::Communicator::*) (const int, const int) )
         & smarties::Communicator::setActionOptions,
         py::arg("n_options"), py::arg("agentID") = 0,
         "Set number of discrete control options for agent # 'agentID'.")

    .def("setActionOptions",
         ( void (smarties::Communicator::*) (const std::vector<int>,const int) )
         & smarties::Communicator::setActionOptions,
         py::arg("n_options"), py::arg("agentID") = 0,
         "Set number of discrete control options for agent # 'agentID'.")

    .def("setStateObservable",
         & smarties::Communicator::setStateObservable,
         py::arg("is_observable"), py::arg("agentID") = 0,
         "For each state variable, set whether observed by agent # 'agentID'.")

    .def("setStateScales",
         & smarties::Communicator::setStateScales,
         py::arg("upper_scale"), py::arg("lower_scale"), py::arg("agentID") = 0,
         "Set upper & lower scaling values for the state of agent # 'agentID'.")

    .def("agentsDefineDifferentMDP",
         & smarties::Communicator::agentsDefineDifferentMDP,
         "Specify that each agent defines a different MPD (state/action/rew).")

    .def("disableDataTrackingForAgents",
         & smarties::Communicator::disableDataTrackingForAgents,
         py::arg("agentStart"), py::arg("agentEnd"),
         "Set agents whose experiences should not be used as training data.")

    .def("isTraining",
         & smarties::Communicator::isTraining,
         "Returns true if smarties is training, false if evaluating a policy.")

    .def("terminateTraining",
         & smarties::Communicator::terminateTraining,
         "Returns true if smarties is requesting application to exit.")

    .def("setNumAppendedPastObservations",
        & smarties::Communicator::setNumAppendedPastObservations,
        py::arg("n_appended"), py::arg("agentID") = 0,
        "Specify that the state of agent # 'agentID' should be composed with "
        "the current observation along with n_appended past ones. "
        "Like it was done in the Atari Nature paper to avoid using RNN.")

    .def("setIsPartiallyObservable",
        & smarties::Communicator::setIsPartiallyObservable,
        py::arg("agentID") = 0,
        "Specify that the decision process of agent # 'agentID' is "
        "non-Markovian and therefore smarties will use RNN.")

    .def("setPreprocessingConv2d",
        & smarties::Communicator::setPreprocessingConv2d,
        py::arg("input_width"), py::arg("input_height"), py::arg("input_features"),
        py::arg("kernels_num"), py::arg("filters_size"), py::arg("stride"),
        py::arg("agentID") = 0,
        "Request a convolutional layer in preprocessing of state for "
        "agent # 'agentID'. This function can be called multiple times to "
        "add multiple conv2d layers, but sizes (widths, heights, filters) must "
        "be consistent otherwise it will trigger an abort.");
}
