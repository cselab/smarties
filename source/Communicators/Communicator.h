//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include <sstream>
#include <sys/un.h>
#ifdef __RL_MPI_CLIENT
#include <mpi.h>
#endif
#ifdef __Smarties_
#include "../Settings.h"
#endif

#include <vector>
#include <cstring>
#include <random>

#define envInfo  int
#define CONT_COMM 0
#define INIT_COMM 1
#define TERM_COMM 2
#define TRNC_COMM 3
#define FAIL_COMM 4
#define _AGENT_KILLSIGNAL -256

#ifdef OPEN_MPI
#define MPI_INCLUDED
#endif

#include "Communicator_utils.h"

class Communicator
{
 public:
  #ifdef MPI_INCLUDED
    MPI_Comm comm_inside_app = MPI_COMM_NULL, comm_learn_pool = MPI_COMM_NULL;
  #endif
  // only for MPI-based *applications* eg. flow solvers:
  int rank_inside_app = -1, rank_learn_pool = -1;
  // comm to talk to master:
  int size_inside_app = -1, size_learn_pool = -1;
  // for MPI-based applications, to split simulations between groups of ranks
  int workerGroup = -1;
  // should be named nState/ActionComponents
  int nStates = -1, nActions = -1;
  int getStateDim()  {return nStates;}
  int getActionDim() {return nActions;}

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
  bool called_by_app = false, spawner = false;
  std::vector<std::vector<double>> stored_actions;
  //internal counters
  unsigned long seq_id = 0, msg_id = 0, iter = 0;

  std::mt19937 gen;

  bool sentStateActionShape = false;
  std::vector<double> obs_bounds, obs_inuse, action_options, action_bounds;

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
  std::vector<double> recvAction(const int iAgent = 0)
  {
    return stored_actions[iAgent];
  }

  void launch();

  Communicator(const int socket, const int state_components, const int action_components, const int number_of_agents = 1);
  Communicator(const int socket, const bool spawn);

  #ifdef MPI_INCLUDED
  Communicator(const int socket, const int state_components, const int action_components, const MPI_Comm app, const int number_of_agents);
  #endif

  virtual ~Communicator();

 protected:
  //App output file descriptor
  int fd;
  fpos_t pos;

  //Communication over sockets
  int socket_id, Socket, ServerSocket;
  char SOCK_PATH[256];
  struct sockaddr_un serverAddress, clientAddress;

  void print();

  void update_rank_size()
  {
    #ifdef MPI_INCLUDED
    if (comm_inside_app != MPI_COMM_NULL) {
      MPI_Comm_rank(comm_inside_app, &rank_inside_app);
      MPI_Comm_size(comm_inside_app, &size_inside_app);
    }
    if (comm_learn_pool != MPI_COMM_NULL) {
      MPI_Comm_rank(comm_learn_pool, &rank_learn_pool);
      MPI_Comm_size(comm_learn_pool, &size_learn_pool);
    }
    #endif
  }

  virtual void launch_forked();
  void setupClient();
  void setupServer();

  void sendStateActionShape();


  #ifdef MPI_INCLUDED
    MPI_Request send_request = MPI_REQUEST_NULL;
    MPI_Request recv_request = MPI_REQUEST_NULL;

    void workerRecv_MPI() {
      //auto start = std::chrono::high_resolution_clock::now();
      assert(comm_learn_pool != MPI_COMM_NULL);
      assert(recv_request != MPI_REQUEST_NULL);
      //if(recv_request != MPI_REQUEST_NULL) {
        while(true) {
          int completed=0;
          MPI_Test(&recv_request, &completed, MPI_STATUS_IGNORE);
          if (completed) break;
          usleep(1);
        }
      //  memcpy(&stored_actions[iAgent], data_action, size_action);
      //}
      assert(recv_request == MPI_REQUEST_NULL);
      //auto elapsed = std::chrono::high_resolution_clock::now() - start;
      //cout << chrono::duration_cast<chrono::microseconds>(elapsed).count() <<endl;
    }

    void workerSend_MPI() {
      //if(send_request != MPI_REQUEST_NULL) MPI_Wait(&send_request, MPI_STATUS_IGNORE);
      //if(recv_request != MPI_REQUEST_NULL) workerRecv_MPI(iAgent);

      assert(comm_learn_pool != MPI_COMM_NULL);
      MPI_Request dummyreq;
      MPI_Isend(data_state, size_state, MPI_BYTE, 0, 1, comm_learn_pool, &dummyreq);
      MPI_Request_free(&dummyreq); //Not my problem? send_request
      MPI_Irecv(data_action, size_action, MPI_BYTE, 0, 0, comm_learn_pool, &recv_request);
    }
  #endif
};
