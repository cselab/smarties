//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Communicator_internal.h"

#include <regex>
#include <algorithm>
#include <iterator>

static inline vector<string> split(const string &s, const char delim) {
  stringstream ss(s); string item; vector<string> tokens;
  while (getline(ss, item, delim)) tokens.push_back(item);
  return tokens;
}

int app_main(Communicator*const rlcom, MPI_Comm mpicom, int argc, char**argv, const Uint numSteps);

Communicator_internal::Communicator_internal(MPI_Comm scom, int sid, bool spawn,
  Settings&sett) : Communicator(sid, spawn, sett.generators[0],
  sett.bTrain, sett.totNumSteps) {
  comm_learn_pool = scom;
  update_rank_size();
}

Communicator_internal::~Communicator_internal() {
  if (rank_learn_pool>0) {
    data_action[0] = AGENT_KILLSIGNAL;
    send_all(Socket, data_action, size_action);
  }
}

void Communicator_internal::launch_forked() {
  assert(not called_by_app);
  createGo_rundir();
  redirect_stdout_init();
  launch_exec("../"+execpath, socket_id);
}

void Communicator_internal::createGo_rundir() {
  char newd[1024];
  getcwd(initd, 512);
  struct stat fileStat;
  while(true) {
    const int workID = workerGroup>=0? workerGroup : socket_id;
    sprintf(newd,"%s/%s_%03d_%05lu", initd, "simulation", workID, iter);
    if ( stat(newd, &fileStat) >= 0 ) iter++; // directory already exists
    else {
      if(rank_inside_app>=0) MPI_Barrier(comm_inside_app);
      if(rank_inside_app<=0) // app's root sets up working dir
        mkdir(newd, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      if(rank_inside_app>=0) MPI_Barrier(comm_inside_app);
      chdir(newd);
      break;
    }
  }
}

int Communicator_internal::recvStateFromApp() {
  int bytes = recv_all(Socket, data_state, size_state);

  if (bytes <= 0)
  {
    if (bytes == 0) printf("socket %d hung up\n", Socket);
    else perror("(1) recv");
    close(Socket);

    intToDoublePtr(0, data_state+0);
    intToDoublePtr(FAIL_COMM, data_state+1);
    iter++;
  }
  else assert(bytes == size_state);

  if(comm_learn_pool != MPI_COMM_NULL) workerSend_MPI();

  return bytes <= 0;
}

int Communicator_internal::sendActionToApp() {
  //printf("I think im sending action %f\n",data_action[0]);
  if(comm_learn_pool != MPI_COMM_NULL) workerRecv_MPI();

  bool endSignal = fabs(data_action[0]-AGENT_KILLSIGNAL)<2.2e-16;

  send_all(Socket, data_action, size_action);

  return endSignal;
}

void Communicator_internal::answerTerminateReq(const double answer) {
  data_action[0] = answer;
   //printf("I think im givign the goahead %f\n",data_action[0]);
  send_all(Socket, data_action, size_action);
}

void Communicator_internal::ext_app_run() {
  assert(workerGroup>=0 && rank_inside_app>=0 &&comm_inside_app!=MPI_COMM_NULL);
  vector<string> argsFiles = split(paramfile, ',');
  vector<string> stepNmbrs = split(nStepPerFile, ',');
  if(argsFiles.size() not_eq stepNmbrs.size())
    die("error reading settings: nStepPappSett and appSettings");
  if(argsFiles.size() == 0) {
    if(paramfile not_eq "") die("");
    argsFiles.push_back("");
  }
  vector<Uint> stepPrefix(argsFiles.size(), 0);
  for (size_t i=1; i<stepNmbrs.size(); i++)
    stepPrefix[i] = stepPrefix[i-1] + std::stol(stepNmbrs[i-1]);
  stepPrefix.push_back(numeric_limits<Uint>::max()); //last setup used for ever
  assert(stepPrefix.size() == argsFiles.size() + 1);

  while(1)
  {
    createGo_rundir();
    if (rank_inside_app==0 && setupfolder != "") //copy any additional file
      if (copy_from_dir(("../"+setupfolder).c_str()) !=0 )
        _die("Error in copy from dir %s\n", setupfolder.c_str());
    MPI_Barrier(comm_inside_app);

    // app only needs lower level functionalities:
    // ie. send state, recv action, specify state/action spaces properties...
    Communicator* const commptr = static_cast<Communicator*>(this);
    Uint settingsInd = 0;
    for(size_t i=0; i<argsFiles.size(); i++)
      if(learner_step_id >= stepPrefix[i]) settingsInd = i;
    Uint numStepTSet = stepPrefix[settingsInd+1] - learner_step_id;
    numStepTSet = numStepTSet / (size_learn_pool-1);
    vector<char*> args = readRunArgLst(argsFiles[settingsInd]);

    redirect_stdout_init();
    app_main(commptr, comm_inside_app, args.size()-1, args.data(), numStepTSet);
    redirect_stdout_finalize();

    for(size_t i = 0; i < args.size()-1; i++) delete[] args[i];
    chdir(initd);  // go up one level
  }
}

vector<char*> Communicator_internal::readRunArgLst(const string _paramfile)
{
  std::vector<char*> args;
  if (_paramfile == "") {
    warn("empty parameter file path");
    args.push_back(0);
    return args;
  }
  std::ifstream t(("../"+_paramfile).c_str());
  std::string linestr((std::istreambuf_iterator<char>(t)),
                       std::istreambuf_iterator<char>());
  if(linestr.size() == 0) die("did not find parameter file");
  std::istringstream iss(linestr); // params file is read into iss
  std::string token;
  while(iss >> token) {
    // If one runs an executable and provides an argument like ./exec 'foo bar'
    // then `foo bar' is put in its entirety in argv[1]. However, when we ask
    // user to write a settingsfile, apostrophes are read as characters, not as
    // special symbols, therefore we must do the following workaround to put
    // anything that is written between parenteses in a single argv entry.
    if(token[0]=='\'') {
      token.erase(0, 1); // remove apostrophe ( should have been read as \' )
      std::string continuation;
      while(token.back() not_eq '\'') { // if match apostrophe, we are done
        if(!(iss >> continuation)) die("missing matching apostrophe");
        token += " " + continuation; // add next line to argv entry
      }
      token.erase(token.end()-1, token.end()); // remove trailing apostrophe
    }
    char *arg = new char[token.size() + 1];
    copy(token.begin(), token.end(), arg);  // write into char array
    arg[token.size()] = '\0';
    args.push_back(arg);
  }
  args.push_back(0); // push back nullptr as last entry
  return args; // remember to deallocate it!
}

void Communicator_internal::redirect_stdout_init()
{
  fflush(stdout);
  fgetpos(stdout, &pos);
  fd = dup(fileno(stdout));
  char buf[500];
  int wrank = getRank(MPI_COMM_WORLD);
  sprintf(buf, "output_%03d_%05lu", wrank, iter);
  freopen(buf, "w", stdout);
}

void Communicator_internal::redirect_stdout_finalize()
{
  dup2(fd, fileno(stdout));
  close(fd);
  clearerr(stdout);
  fsetpos(stdout, &pos);        /* for C9X */
}

void Communicator_internal::getStateActionShape()
{
  if(sentStateActionShape) die("undefined behavior");

  double sizes[4] = {0, 0, 0, 0};
  if (rank_learn_pool==0)
    MPI_Recv(sizes, 32, MPI_BYTE, 1, 3, comm_learn_pool, MPI_STATUS_IGNORE);
  else {
    comm_sock(Socket, false, sizes, 4*sizeof(double));
    if(rank_learn_pool==1) MPI_Ssend(sizes,32, MPI_BYTE, 0,3, comm_learn_pool);
  }

  nStates          = doublePtrToInt(sizes+0);
  nActions         = doublePtrToInt(sizes+1);
  discrete_actions = doublePtrToInt(sizes+2);
  nAgents          = doublePtrToInt(sizes+3);
  //printf("Discrete? %d\n",discrete_actions);
  assert(nStates>=0 && nActions>=0);
  update_state_action_dims(nStates, nActions);
  sentStateActionShape = true;

  if (rank_learn_pool==0) {
    MPI_Recv(obs_inuse.data(), nStates*8, MPI_BYTE, 1, 3, comm_learn_pool, MPI_STATUS_IGNORE);
    MPI_Recv(obs_bounds.data(), nStates*16, MPI_BYTE, 1, 4, comm_learn_pool, MPI_STATUS_IGNORE);
    MPI_Recv(action_options.data(), nActions*16, MPI_BYTE, 1, 5, comm_learn_pool, MPI_STATUS_IGNORE);
  } else {
    comm_sock(Socket, false, obs_inuse.data(), nStates*8);
    comm_sock(Socket, false, obs_bounds.data(), nStates*16);
    comm_sock(Socket, false, action_options.data(), nActions*16);
    if (rank_learn_pool==1) {
      MPI_Ssend(obs_inuse.data(), nStates*8, MPI_BYTE, 0, 3, comm_learn_pool);
      MPI_Ssend(obs_bounds.data(), nStates*16, MPI_BYTE, 0, 4, comm_learn_pool);
      MPI_Ssend(action_options.data(), nActions*16, MPI_BYTE, 0, 5, comm_learn_pool);
    }
  }

  int n_vals = 0;
  for(int i=0; i<nActions; i++) n_vals += action_options[i*2];
  discrete_action_values = n_vals;

  action_bounds.resize(n_vals);
  if (rank_learn_pool==0)
    MPI_Recv(action_bounds.data(), n_vals*8, MPI_BYTE, 1, 6, comm_learn_pool, MPI_STATUS_IGNORE);
  else {
    comm_sock(Socket, false, action_bounds.data(), n_vals*8);
    if (rank_learn_pool==1)
      MPI_Ssend(action_bounds.data(),n_vals*8, MPI_BYTE, 0,6, comm_learn_pool);
  }

  print();
}
