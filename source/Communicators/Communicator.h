//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include <vector>
#include <cstring>
#include <random>
#include <sstream>

#define envInfo  int
#define CONT_COMM 0
#define INIT_COMM 1
#define TERM_COMM 2
#define TRNC_COMM 3
#define FAIL_COMM 4
#define AGENT_KILLSIGNAL -256
#define AGENT_TERMSIGNAL  256

#if defined(SMARTIES) || defined(SMARTIES_APP)
#include <mpi.h>
#endif

#include "Communicator_utils.h"

class Communicator
{
 protected:
  // only for MPI-based *applications* eg. flow solvers:
  int rank_inside_app = -1, rank_learn_pool = -1;
  // comm to talk to master:
  int size_inside_app = -1, size_learn_pool = -1;

  // for MPI-based applications, to split simulations between groups of ranks
  // each learner can have multiple mpi groups of workers
  int workerGroup = -1;
  // should be named nState/ActionComponents
  int nStates = -1, nActions = -1;

  // byte size of the messages
  int size_state = -1, size_action = -1;
  // bool whether using discrete act options, number of agents per environment
  int discrete_actions = 0, nAgents=1;

  // number of values contained in action_bounds vector
  // 2*dimA in case of continuous (high and low per each)
  // the number of options in a 1-dimensional discrete action problem
  // if agent must select multiple discrete actions per turn then it should be
  // prod_i ^dimA dimA_i, where dimA_i is number of options for act component i
  int discrete_action_values = 0;
  // communication buffers:
  double *data_state = nullptr, *data_action = nullptr;
  bool called_by_app = false;
  std::vector<std::vector<double>> stored_actions;
  //internal counters
  unsigned long seq_id = 0, msg_id = 0, iter = 0;
  unsigned learner_step_id = 0;
  std::mt19937 gen;

  const bool bTrain;
  const int nEpisodes;
  bool sentStateActionShape = false;
  std::vector<double> obs_bounds, obs_inuse, action_options, action_bounds;

 public:
  std::mt19937& getPRNG();
  bool isTraining();
  int desiredNepisodes();
  int getDimS();
  int getDimA();
  int getNagents();
  int nDiscreteAct();
  std::vector<double>& stateBounds();
  std::vector<double>& isStateObserved();
  std::vector<double>& actionOption();
  std::vector<double>& actionBounds();

  void update_state_action_dims(const int sdim, const int adim);

  // called in user's environment to describe control problem
  void set_action_scales(const std::vector<double> upper,
    const std::vector<double> lower, const bool bound = false);
  void set_action_options(const std::vector<int> options);
  void set_action_options(const int options);
  void set_state_scales(const std::vector<double> upper,
    const std::vector<double> lower);
  void set_state_observable(const std::vector<bool> observable);

  //called by app to interact with smarties
  void sendState(const int iAgent, const envInfo status,
    const std::vector<double> state, const double reward);

  // specialized functions:
  // initial state: the first of an ep. By definition 0 reward (no act done yet)
  inline void sendInitState(const std::vector<double> state, const int iAgent=0)
  {
    return sendState(iAgent, INIT_COMM, state, 0);
  }
  // terminal state: the last of a sequence which ends because a TERMINAL state
  // has been encountered (ie. agent cannot continue due to failure)
  inline void sendTermState(const std::vector<double> state, const double reward, const int iAgent = 0)
  {
    return sendState(iAgent, TERM_COMM, state, reward);
  }
  // truncation: usually episode can be over due to time constrains
  // it differs because the policy was not at fault for ending the episode
  // meaning that episode could continue following the policy
  // and that the value of this last state should be the expected on-pol returns
  inline void truncateSeq(const std::vector<double> state, const double reward, const int iAgent = 0)
  {
    return sendState(iAgent, TRNC_COMM, state, reward);
  }
  // `normal` state inside the episode
  inline void sendState(const std::vector<double> state, const double reward, const int iAgent = 0)
  {
    return sendState(iAgent, CONT_COMM, state, reward);
  }
  // receive action sent by smarties
  std::vector<double> recvAction(const int iAgent = 0);

  Communicator(const int socket, const int state_components, const int action_components, const int number_of_agents = 1);
  Communicator(int socket, bool spawn, std::mt19937& G, int _bTr, int nEps);
  virtual ~Communicator();

  virtual void launch();

 protected:
  //App output file descriptor
  int fd;
  fpos_t pos;

  //Communication over sockets
  int socket_id, Socket;

  void print();

  void update_rank_size();

  void sendStateActionShape();

 #ifdef MPI_VERSION
    MPI_Request send_request = MPI_REQUEST_NULL;
    MPI_Request recv_request = MPI_REQUEST_NULL;

    void workerRecv_MPI();
    void workerSend_MPI();

    MPI_Comm comm_inside_app = MPI_COMM_NULL, comm_learn_pool = MPI_COMM_NULL;

 public:

    Communicator(const int socket, const int state_comp, const int action_comp,
      const MPI_Comm app, const int number_of_agents);
 #endif
};
