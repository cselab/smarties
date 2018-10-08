//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Communicator.h"
#include "Communicator_utils.cpp"
#include <iomanip>
#include <fstream>
#include <iostream>

//APPLICATION SIDE CONSTRUCTOR
Communicator::Communicator(int socket, int state_comp, int action_comp,
  int number_of_agents) : gen(socket), bTrain(true), nEpisodes(-1)
{
  if(socket<0) {
    printf("FATAL: Communicator created with socket < 0.\n");
    abort();
  }
  if(state_comp<=0) {
    printf("FATAL: Cannot set negative state space dimensionality.\n");
    abort();
  }
  if(action_comp<=0) {
    printf("FATAL: Cannot set negative number of action degrees of freedom.\n");
    abort();
  }
  if(number_of_agents<=0) {
    printf("FATAL: Cannot set negative number of agents.\n");
    abort();
  }
  assert(state_comp>0 && action_comp>0 && number_of_agents>0);
  nAgents = number_of_agents;
  update_state_action_dims(state_comp, action_comp);
  spawner = socket==0; // if app gets socket prefix 0, then it spawns smarties
  socket_id = socket;
  called_by_app = true;
  launch();
}

#ifdef MPI_VERSION //MPI APPLICATION SIDE CONSTRUCTOR
  Communicator::Communicator(const int socket, const int state_components,
  const int action_components, const MPI_Comm app, const int number_of_agents)
  : Communicator(socket, state_components, action_components, number_of_agents)
  {
    comm_inside_app = app;
    update_rank_size();
  }
#endif

// this function effectively sends the state of agent iAgent
// (and additional info) and also waits for and stores selected action
void Communicator::sendState(const int iAgent, const envInfo status,
    const std::vector<double> state, const double reward)
{
  if(rank_inside_app <= 0)
  { //only rank 0 of the app sends state
    if(!sentStateActionShape) sendStateActionShape();
    assert(state.size()==(std::size_t)nStates && data_state not_eq nullptr);
    assert(iAgent>=0 && iAgent<nAgents);

    intToDoublePtr(iAgent, data_state+0);
    intToDoublePtr(status, data_state+1);
    for (int j=0; j<nStates; j++) {
      data_state[j+2] = state[j];
      assert(not std::isnan(state[j]) && not std::isinf(state[j]));
    }
    data_state[nStates+2] = reward;
    assert(not std::isnan(reward) && not std::isinf(reward));

    #ifdef MPI_VERSION
      if (rank_learn_pool>0) workerSend_MPI();
      else
    #endif
      comm_sock(Socket, true, data_state, size_state);
  }

  // Now prepare next action for the same agent:
  #ifdef MPI_VERSION
    if(rank_inside_app <= 0)
    {
      if (rank_learn_pool>0) workerRecv_MPI();
      else
  #endif
        comm_sock(Socket, false, data_action, size_action);

  #ifdef MPI_VERSION // app rank 0 sends to other ranks
      for (int i=1; i<size_inside_app; ++i)
        MPI_Send(data_action, size_action, MPI_BYTE, i, 42, comm_inside_app);
    } else {
      MPI_Recv(data_action, size_action, MPI_BYTE, 0, 42, comm_inside_app, MPI_STATUS_IGNORE);
    }
  #endif

  if(std::fabs(data_action[0]-AGENT_KILLSIGNAL)<2.2e-16) abort();

  if (status >= TERM_COMM) {
    seq_id++;
    msg_id = 0;
    learner_step_id = (unsigned) * data_action;
    stored_actions[iAgent][0] = AGENT_TERMSIGNAL;
  } else {
    for (int j=0; j<nActions; j++) {
      stored_actions[iAgent][j] = data_action[j];
      assert(not std::isnan(data_action[j]) && not std::isinf(data_action[j]));
    }
  }
}

std::vector<double> Communicator::recvAction(const int iAgent)
{
  assert( std::fabs(stored_actions[iAgent][0]-AGENT_TERMSIGNAL) > 2.2e-16 );
  return stored_actions[iAgent];
}

void Communicator::set_action_scales(const std::vector<double> upper,
  const std::vector<double> lower, const bool bound)
{
  assert(!sentStateActionShape);
  discrete_actions = 0;
  assert(upper.size() == (size_t)nActions && lower.size() == (size_t)nActions);
  for (int i=0; i<nActions; i++) action_bounds[2*i+0] = upper[i];
  for (int i=0; i<nActions; i++) action_bounds[2*i+1] = lower[i];
  for (int i=0; i<nActions; i++) action_options[2*i+0] = 2.1;
  for (int i=0; i<nActions; i++) action_options[2*i+1] = bound ? 1.1 : 0;
}

void Communicator::set_action_options(const std::vector<int> action_option_num)
{
  assert(!sentStateActionShape);
  discrete_actions = 1;
  assert(action_option_num.size() == (size_t)nActions);
  discrete_action_values = 0;
  for (int i=0; i<nActions; i++) {
    discrete_action_values += action_option_num[i];
    action_options[2*i+0] = action_option_num[i];
    action_options[2*i+1] = 1.1;
  }

  action_bounds.resize(discrete_action_values);
  for(int i=0, k=0; i<nActions; i++)
    for(int j=0; j<action_option_num[i]; j++)
      action_bounds[k++] = j;
}

void Communicator::set_action_options(const int action_option_num)
{
  assert(!sentStateActionShape);
  if(nActions != 1) {
    printf("FATAL: Communicator::set_action_options perceived more than 1 action degree of freedom, but only one number of actions provided.\n");
    abort();
  }
  assert(1 == nActions);
  discrete_actions = 1;
  discrete_action_values = action_option_num;
  action_options[0] = action_option_num;
  action_options[1] = 1.1;

  action_bounds.resize(action_option_num);
  for(int j=0; j<action_option_num; j++) action_bounds[j] = j;
}

void Communicator::set_state_scales(const std::vector<double> upper,
  const std::vector<double> lower)
{
  assert(!sentStateActionShape);
  assert(upper.size() == (size_t)nStates && lower.size() == (size_t)nStates);
  for (int i=0; i<nStates; i++) obs_bounds[2*i+0] = upper[i];
  for (int i=0; i<nStates; i++) obs_bounds[2*i+1] = lower[i];
}

void Communicator::set_state_observable(const std::vector<bool> observable)
{
  assert(!sentStateActionShape);
  assert(observable.size() == (size_t) nStates);
  for (int i=0; i<nStates; i++) obs_inuse[i] = observable[i];
}

void Communicator::sendStateActionShape()
{
  if(sentStateActionShape) return;
  sentStateActionShape = true;

  assert(obs_inuse.size() == (size_t) nStates);
  assert(obs_bounds.size() == (size_t) nStates*2);
  assert(action_bounds.size() == (size_t) nActions*2);
  assert(action_options.size() == (size_t) discrete_action_values);

  // only rank 0 of MPI-based apps send the info to smarties:
  if(rank_inside_app > 0) return;
  if(workerGroup > 0) return;

  double sizes[4] = {nStates+.1, nActions+.1, discrete_actions+.1, nAgents+.1};
  #ifdef MPI_VERSION
    if (rank_learn_pool>0) {
      MPI_Ssend(sizes, 4*8, MPI_BYTE, 0,3, comm_learn_pool);
      MPI_Ssend(obs_inuse.data(), nStates*1*8, MPI_BYTE, 0, 3, comm_learn_pool);
      MPI_Ssend(obs_bounds.data(), nStates*16, MPI_BYTE, 0, 4, comm_learn_pool);
      MPI_Ssend(action_options.data(), nActions*16, MPI_BYTE, 0, 5, comm_learn_pool);
      MPI_Ssend(action_bounds.data(), discrete_action_values*8, MPI_BYTE, 0, 6, comm_learn_pool);
    } else
  #endif
    {
      comm_sock(Socket, true, sizes, 4 *sizeof(double));
      comm_sock(Socket, true, obs_inuse.data(),      nStates *1*sizeof(double));
      comm_sock(Socket, true, obs_bounds.data(),     nStates *2*sizeof(double));
      comm_sock(Socket, true, action_options.data(), nActions*2*sizeof(double));
      comm_sock(Socket, true, action_bounds.data(),  discrete_action_values*8 );
    }

  print();
}

void Communicator::update_state_action_dims(const int sdim, const int adim)
{
  if(sentStateActionShape) {
    assert(adim==nActions);
    assert(sdim==nStates);
    return;
  }
  assert(adim>0);
  assert(sdim>0);
  assert(nAgents>0);
  nStates = sdim;
  nActions = adim;
  discrete_action_values = 2*adim;
  obs_inuse      = std::vector<double>(1*sdim, 1);
  obs_bounds     = std::vector<double>(2*sdim, 0);
  action_options = std::vector<double>(2*adim, 0);
  action_bounds  = std::vector<double>(2*adim, 0);
  for (int i = 0; i<2*sdim; i++) obs_bounds[i]     = i%2 == 0 ? 1   : -1;
  for (int i = 0; i<2*adim; i++) action_options[i] = i%2 == 0 ? 2.1 :  0;
  for (int i = 0; i<2*adim; i++) action_bounds[i]  = i%2 == 0 ? 1   : -1;
  // agent number, initial/normal/terminal indicator, state,  reward
  stored_actions = std::vector<std::vector<double>>(nAgents, std::vector<double>(nActions,0));
  size_state = (3+sdim)*sizeof(double);
  size_action = adim*sizeof(double);
  _dealloc(data_action);
  _dealloc(data_state);
  data_action = _alloc(size_action);
  data_state = _alloc(size_state);
}

void Communicator::launch_forked()
{
  printf("disabled Client style scripts"); fflush(0); abort();
  assert(called_by_app);
  //go up til a file runClient is found: shaky
  struct stat buffer;
  while(stat("runClient.sh", &buffer)) {
    chdir("..");
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL)
      printf("Current working dir: %s\n", cwd);
    else perror("getcwd() error");
  }

  fd = redirect_stdout_stderr();
  launch_exec("./runClient.sh", socket_id);
}

void Communicator::launch()
{
  assert(rank_inside_app<1);
  if (spawner && called_by_app) {
    //this code runs if application spawns smarties
    //cheap way to ensure multiple sockets can exist on same node
    struct timeval clock;
    gettimeofday(&clock, NULL);
    socket_id = abs(clock.tv_usec % std::numeric_limits<int>::max());
  }
  sprintf(SOCK_PATH, "%s%d", "/tmp/smarties_sock", socket_id);

  if (spawner) setupClient();
  else setupServer();
}

void Communicator::setupClient()
{
  unlink(SOCK_PATH);
  const int rf = fork();

  if (rf == 0) //child spawns process
  {
    launch_forked();
    printf("setupClient app returned: what TODO?"); fflush(0); abort();
  }
  else //parent
  {
    Socket = socket(AF_UNIX, SOCK_STREAM, 0);

    int _true = 1;
    if(setsockopt(Socket, SOL_SOCKET, SO_REUSEADDR, &_true, sizeof(int))<0) {
      printf("Sockopt failed\n"); fflush(0); abort();
    }

    // Specify the server
    bzero((char *)&serverAddress, sizeof(serverAddress));
    serverAddress.sun_family = AF_UNIX;
    strcpy(serverAddress.sun_path, SOCK_PATH);
    const int servlen = sizeof(serverAddress.sun_family)
                      + strlen(serverAddress.sun_path)+1;

    // Connect to the server
    while (connect(Socket, (struct sockaddr *)&serverAddress, servlen) < 0)
      usleep(1);
  }
}

void Communicator::setupServer()
{
  if ((ServerSocket = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
    printf("socket"); fflush(0); abort();
  }

  bzero(&serverAddress, sizeof(serverAddress));
  serverAddress.sun_family = AF_UNIX;
  strcpy(serverAddress.sun_path, SOCK_PATH);
  //this printf is to check that there is no funny business with trailing 0s:
  //printf("%s %s\n",serverAddress.sun_path, SOCK_PATH); fflush(0);
  const int servlen = sizeof(serverAddress.sun_family)
                    + strlen(serverAddress.sun_path) +1;

  if (bind(ServerSocket, (struct sockaddr *)&serverAddress, servlen) < 0) {
    printf("bind"); fflush(0); abort();
  }
  /*
  int _true = 1;
  if(setsockopt(ServerSocket, SOL_SOCKET, SO_REUSEADDR, &_true, sizeof(int))<0)
  {
    perror("Sockopt failed\n");
    exit(1);
  }
  */

  if (listen(ServerSocket, 1) == -1) { // listen (only 1)
    printf("listen"); fflush(0); abort();
  }

  unsigned int addr_len = sizeof(clientAddress);
  struct sockaddr* const clientAddrPtr = (struct sockaddr*) &clientAddress;
  if( (Socket=accept(ServerSocket, clientAddrPtr, &addr_len)) == -1) {
    printf("accept"); fflush(0); abort();
  }

  printf("selectserver: new connection from on socket %d\n", Socket);
  fflush(0);
}

Communicator::~Communicator()
{
  if (rank_learn_pool>0) {
    if (spawner) close(Socket);
    else   close(ServerSocket);
  } //if with forked process paradigm
  if(data_state not_eq nullptr)  _dealloc(data_state);
  if(data_action not_eq nullptr) _dealloc(data_action);
}

// ONLY FOR CHILD CLASS
Communicator::Communicator(int socket, bool spawn, std::mt19937& G, int _bTr,
  int nEps) : gen(G()), bTrain(_bTr), nEpisodes(nEps)
{
  if(socket<0) {
    printf("FATAL: Communicator created with socket < 0.\n");
    abort();
  }
  spawner = spawn; // if app gets socket prefix 0, then it spawns smarties
  socket_id = socket;
}

void Communicator::print()
{
  std::ostringstream fname;
  int wrank = socket_id;
  #ifdef MPI_VERSION
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
  #endif
  fname<<"comm_"<<std::setw(3)<<std::setfill('0')<<wrank<<".log";
  std::ofstream o(fname.str().c_str(), std::ios::app);
  o <<(spawner?"Server":"Client")<<" communicator on ";
  o <<(called_by_app?"app":"smarties")<<" side:\n";
  o <<"nStates:"<<nStates<<" nActions:"<<nActions;
  o <<" size_action:"<<size_action<< " size_state:"<< size_state<<"\n";
  o <<"MPI comm: size_s"<<size_learn_pool<<" rank_s:"<<rank_learn_pool;
  o <<" size_a:"<<size_inside_app<< " rank_a:"<< rank_inside_app<<"\n";
  //o <<"Socket comm: prefix:"<<socket_id<<" PATH:"<<std::string(SOCK_PATH)<<"\n";
  o.close();
}

void Communicator::update_rank_size()
{
  #ifdef MPI_VERSION
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

#ifdef MPI_VERSION
void Communicator::workerSend_MPI() {
  //if(send_request != MPI_REQUEST_NULL) MPI_Wait(&send_request, MPI_STATUS_IGNORE);
  //if(recv_request != MPI_REQUEST_NULL) workerRecv_MPI(iAgent);

  assert(comm_learn_pool != MPI_COMM_NULL);
  MPI_Request dummyreq;
  MPI_Isend(data_state, size_state, MPI_BYTE, 0, 1, comm_learn_pool, &dummyreq);
  MPI_Request_free(&dummyreq); //Not my problem? send_request
  MPI_Irecv(data_action, size_action, MPI_BYTE, 0, 0, comm_learn_pool, &recv_request);
}
void Communicator::workerRecv_MPI() {
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
#endif

std::mt19937& Communicator::getPRNG() {
  return gen;
}
bool Communicator::isTraining() {
  return bTrain;
}
int Communicator::desiredNepisodes() {
  return nEpisodes;
}
int Communicator::getDimS() {
  return nStates;
}
int Communicator::getDimA() {
  return nActions;
}
int Communicator::getNagents() {
  return nAgents;
}
int Communicator::nDiscreteAct() {
  return discrete_actions;
}
std::vector<double>& Communicator::stateBounds() {
  return obs_bounds;
}
std::vector<double>& Communicator::isStateObserved() {
  return obs_inuse;
}
std::vector<double>& Communicator::actionOption() {
  return action_options;
}
std::vector<double>& Communicator::actionBounds() {
  return action_bounds;
}
