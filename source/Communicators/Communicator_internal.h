//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "Communicator.h"
#include "../Settings.h"

class Communicator_internal: public Communicator
{
 protected:
  const Settings& S;
  char initd[512];
  std::string execpath    = S.launchfile;
  std::string paramfile   = S.appSettings;
  std::string nStepPerFile= S.nStepPappSett;
  std::string setupfolder = S.setupFolder;
  const int nOwnWorkers = S.nWorkers_own;
  const int bAsync = S.bAsync;
  const bool bSpawnApp = S.bSpawnApp;

  std::vector<int> clientSockets = std::vector<int>(nOwnWorkers, 0);

  std::mutex& mpi_mutex = S.mpi_mutex;
  std::vector<double*> inpBufs;
  std::vector<double*> outBufs;
  std::vector<MPI_Request> requests =
                        std::vector<MPI_Request>(nOwnWorkers, MPI_REQUEST_NULL);


public:
  void getStateActionShape();
  void launch() override;

  int recvStateFromApp();
  int sendActionToApp();

  double* getDataAction() { return data_action; }
  double* getDataState()  { return data_state; }

  void answerTerminateReq(const double answer);

  void set_application_mpicom(const MPI_Comm acom, const int group)
  {
    comm_inside_app = acom;
    workerGroup = group;
    update_rank_size();
  }

  void restart(std::string fname);
  void save() const;

  void ext_app_run();
  std::vector<char*> readRunArgLst(const std::string _paramfile);
  void redirect_stdout_init();
  void redirect_stdout_finalize();
  void createGo_rundir();

  void sendBuffer(const int i, const std::vector<double> V);

  void recvBuffer(const int i);

  int testBuffer(const int i);

  void sendTerminateReq();

  //called by smarties
  Communicator_internal(Settings& sett);
  ~Communicator_internal();

  static std::vector<double*> alloc_bufs(const int size, const int num);
  void unpackState(const int i, int& agent, envInfo& info,
      std::vector<double>& state, double& reward);
};
