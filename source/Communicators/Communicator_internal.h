//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "Communicator.h"

class Communicator_internal: public Communicator
{
 protected:
  char initd[512];
  std::string execpath    = std::string();
  std::string paramfile   = std::string();
  std::string nStepPerFile= std::string();
  std::string setupfolder = std::string();

  void launch_forked() override;

public:
  void getStateActionShape();

  int recvStateFromApp();
  int sendActionToApp();

  double* getDataAction() { return data_action; }
  double* getDataState()  { return data_state; }

  void answerTerminateReq(const double answer);

  void set_params_file(const std::string fname) { paramfile = fname; }
  void set_nstepp_file(const std::string fname) { nStepPerFile = fname; }

  void set_exec_path(const std::string fname) { execpath = fname; }
  void set_folder_path(const std::string fname) { setupfolder = fname; }
  void set_application_mpicom(const MPI_Comm acom, const int group)
  {
    comm_inside_app = acom;
    workerGroup = group;
    update_rank_size();
  }

  void restart(std::string fname);
  void save() const;

  void ext_app_run();
  vector<char*> readRunArgLst(const string _paramfile);
  void redirect_stdout_init();
  void redirect_stdout_finalize();
  void createGo_rundir();
  //called by smarties
  Communicator_internal(const MPI_Comm scom, const int socket, const bool spawn);
  ~Communicator_internal();
};

inline void unpackState(double* const data, int& agent, envInfo& info,
    std::vector<double>& state, double& reward)
{
  assert(data not_eq nullptr);
  agent = doublePtrToInt(data+0);
  info  = doublePtrToInt(data+1);
  for (unsigned j=0; j<state.size(); j++) {
    state[j] = data[j+2];
    assert(not std::isnan(state[j]));
    assert(not std::isinf(state[j]));
  }
  reward = data[state.size()+2];
  assert(not std::isnan(reward));
  assert(not std::isinf(reward));
}
