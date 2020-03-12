//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Communicator_h
#define smarties_Communicator_h

#include "Core/Environment.h"
#include "Utils/MPIUtilities.h"
#include <random>

namespace smarties
{
struct COMM_buffer;
class Communicator;
class Worker;
}

// main function callback to user's application
// arguments are: - the communicator with smarties
//                - the mpi communicator to use within the app
//                - argc and argv read from settings file
// wrappers are created if no mpi comm or if no args are needed

using environment_callback_t =
  std::function<void(
      smarties::Communicator*const smartiesCommunicator,
      const MPI_Comm mpiCommunicator,
      int argc, char**argv
    )>;

namespace smarties
{

#define VISIBLE __attribute__((visibility("default")))

class Communicator
{
public:
  Communicator() = delete;
  Communicator(const Communicator&) = delete;

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////////////// BEGINNER METHODS //////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // Send first state of an episode in order to get first action:
  VISIBLE void sendInitState(const std::vector<double>& state,
                             const int agentID=0)
  {
    return _sendState(agentID, INIT, state, 0);
  }

  // Send normal state and reward:
  VISIBLE void sendState(const std::vector<double>& state,
                         const double reward,
                         const int agentID = 0)
  {
    return _sendState(agentID, CONT, state, reward);
  }

  // Send terminal state/reward: the last step of an episode which ends because
  // of TERMINATION (e.g. agent cannot continue due to failure or success).
  VISIBLE void sendTermState(const std::vector<double>& state,
                             const double reward,
                             const int agentID = 0)
  {
    return _sendState(agentID, TERM, state, reward);
  }

  // Send truncated state/reward: the last step of an episode which ends because
  // of TRUNCATION (e.g. agent cannot continue due to time limits). Difference
  // from TERMINATION is that policy was not direct cause of episode's end.
  VISIBLE void sendLastState(const std::vector<double>& state,
                             const double reward,
                             const int agentID = 0)
  {
    return _sendState(agentID, TRNC, state, reward);
  }

  // receive action for the latest given state:
  VISIBLE const std::vector<double> recvAction(const int agentID = 0) const;
  VISIBLE int recvDiscreteAction(const int agentID = 0) const;


  VISIBLE void setNumAgents(int _nAgents);

  VISIBLE void setStateActionDims(const int dimState,
                                  const int dimAct,
                                  const int agentID = 0);

  VISIBLE void setActionScales(const std::vector<double> uppr,
                               const std::vector<double> lowr,
                               const bool bound,
                               const int agentID = 0);

  VISIBLE void setActionScales(const std::vector<double> upper,
                               const std::vector<double> lower,
                               const std::vector<bool>   bound,
                               const int agentID = 0);

  VISIBLE void setActionOptions(const int options,
                                const int agentID = 0);

  VISIBLE void setActionOptions(const std::vector<int> options,
                                const int agentID = 0);

  VISIBLE void setStateObservable(const std::vector<bool> observable,
                                  const int agentID = 0);

  VISIBLE void setStateScales(const std::vector<double> upper,
                              const std::vector<double> lower,
                              const int agentID = 0);

  VISIBLE void setIsPartiallyObservable(const int agentID = 0);

  VISIBLE void finalizeProblemDescription();

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////////////// ADVANCED METHODS //////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  VISIBLE void envHasDistributedAgents();

  VISIBLE void agentsDefineDifferentMDP();

  VISIBLE void disableDataTrackingForAgents(int agentStart, int agentEnd);

  VISIBLE void agentsShareExplorationNoise(const int agentID = 0);

  VISIBLE void setPreprocessingConv2d(
    const int input_width, const int input_height, const int input_features,
    const int kernels_num, const int filters_size, const int stride,
    const int agentID = 0);

  VISIBLE void setNumAppendedPastObservations(const int n_appended,
                                              const int agentID = 0);

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////////// OPTIMIZATION INTERFACE ////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  // conveniency methods for optimization (stateless/timeless) problems

  VISIBLE const std::vector<double> getOptimizationParameters(int agentID = 0)
  {
    assert(ENV.descriptors[agentID]->dimState == 0 &&
           "optimization interface only defined for stateless problems");
    _sendState(agentID, INIT, std::vector<double>(0), 0); // fake initial state
    return recvAction(agentID);
  }

  VISIBLE void setOptimizationEvaluation(const Real R, const int agentID = 0)
  {
    assert(ENV.descriptors[agentID]->dimState == 0 &&
           "optimization interface only defined for stateless problems");
    _sendState(agentID, TERM, std::vector<double>(0), R); // send objective eval
  }

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////////////// UTILITY METHODS ///////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  VISIBLE std::mt19937& getPRNG();
  VISIBLE Real getUniformRandom(const Real begin = 0, const Real end = 1);
  VISIBLE Real getNormalRandom(const Real mean = 0, const Real stdev = 1);

  VISIBLE unsigned getLearnersGradStepsNum(const int agentID = 0);
  VISIBLE unsigned getLearnersTrainingTimeStepsNum(const int agentID = 0);
  VISIBLE double getLearnersAvgCumulativeReward(const int agentID = 0);

  VISIBLE bool isTraining() const;

  VISIBLE bool terminateTraining() const;

  //////////////////////////////////////////////////////////////////////////////
  ///////////////////////////// DEVELOPER METHODS //////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

protected:
  bool bEnvDistributedAgents = false;

  Environment ENV;
  std::vector<std::unique_ptr<Agent>>& agents = ENV.agents;
  std::vector<std::unique_ptr<COMM_buffer>> BUFF;

  struct {
    int server = -1;
    std::vector<int> clients;
  } SOCK;

  void synchronizeEnvironments();
  void initOneCommunicationBuffer();
  //random number generation:
  std::mt19937 gen;
  //internal counters & flags
  bool bTrain = true;
  bool bTrainIsOver = false;
  long nRequestedEnvTimeSteps = -1;
  Uint globalTstepCounter = 0;

  //called by app to interact with smarties:
  VISIBLE void _sendState(const int agentID,
                          const episodeStatus status,
                          const std::vector<double>& state,
                          const double reward);

  //access to smarties' internals, available only if app is linked into exec
  friend class Worker;

  Worker * const worker = nullptr;

  Communicator(Worker* const, std::mt19937&, bool);
};

#undef VISIBLE

struct COMM_buffer
{
  COMM_buffer(const size_t maxSdim, const size_t maxAdim) :
    maxStateDim(maxSdim), maxActionDim(maxAdim),
    sizeStateMsg(Agent::computeStateMsgSize(maxSdim)),
    sizeActionMsg(Agent::computeActionMsgSize(maxAdim)),
    dataStateBuf (malloc(sizeStateMsg) ), // aligned_alloc(1024...)
    dataActionBuf(malloc(sizeActionMsg)) { }

  ~COMM_buffer() {
    assert(dataStateBuf not_eq nullptr && dataActionBuf not_eq nullptr);
    free(dataActionBuf);
    free(dataStateBuf);
  }

  COMM_buffer(const COMM_buffer& c) = delete;
  COMM_buffer& operator= (const COMM_buffer& s) = delete;

  const size_t maxStateDim, maxActionDim, sizeStateMsg, sizeActionMsg;
  void * const dataStateBuf;
  void * const dataActionBuf;
};

} // end namespace smarties
#endif // smarties_Communicator_h
