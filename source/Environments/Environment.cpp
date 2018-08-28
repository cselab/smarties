//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Environment.h"
#include "../Network/Builder.h"

Environment::Environment(Settings& _settings) :
g(&_settings.generators[0]), settings(_settings), gamma(_settings.gamma) {}

void Environment::setDims()
{
  comm_ptr->getStateActionShape();
  assert(comm_ptr->nAgents>0);
  assert(comm_ptr->nStates>0);
  assert(comm_ptr->nActions>0);
  assert(comm_ptr->discrete_actions>=0);

  sI.dim = comm_ptr->nStates;
  aI.dim = comm_ptr->nActions;
  nAgentsPerRank = comm_ptr->nAgents;
  aI.discrete = comm_ptr->discrete_actions;

  aI.values.resize(aI.dim);
  aI.bounded.resize(aI.dim, 0);
  sI.mean.resize(sI.dim);
  sI.scale.resize(sI.dim);
  sI.inUse.resize(sI.dim, 1);

  if(!settings.world_rank) printf("State dimensionality : %d.",sI.dim);
  for (unsigned i=0; i<sI.dim; i++) {
    const bool inuse = comm_ptr->obs_inuse[i] > 0.5;
    const double upper = comm_ptr->obs_bounds[i*2+0];
    const double lower = comm_ptr->obs_bounds[i*2+1];
    sI.inUse[i] = inuse;
    sI.mean[i]  = 0.5*(upper+lower);
    sI.scale[i] = 0.5*std::fabs(upper-lower);
    if(sI.scale[i]>=1e3 || sI.scale[i] < 1e-7) {
      if(settings.world_rank == 0) printf(" unbounded");
      sI.scale = Rvec(sI.dim, 1); sI.mean = Rvec(sI.dim, 0);
      break;
    }
    if(!settings.world_rank && not inuse)
      printf(" State component %u is hidden from the learner.", i);
  }
  if(!settings.world_rank) printf("\nAction dimensionality : %d.",aI.dim);

  int k = 0;
  for (Uint i=0; i<aI.dim; i++) {
    aI.bounded[i]   = comm_ptr->action_options[i*2 +1] > 0.5;
    const int nvals = comm_ptr->action_options[i*2 +0];
    // if act space is continuous, only receive high and low val for each action
    assert(aI.discrete || nvals == 2);
    assert(nvals > 1);
    aI.values[i].resize(nvals);
    for(int j=0; j<nvals; j++)
      aI.values[i][j] = comm_ptr->action_bounds[k++];

    const Real amax = aI.getActMaxVal(i), amin = aI.getActMinVal(i);
    if(!settings.world_rank)
    printf(" [%u: %f:%f%s]", i, amin, amax, aI.bounded[i]?" (bounded)":"");
  }
  if(!settings.world_rank) printf("\n");

  commonSetup(); //required
  assert(sI.dim == (Uint) comm_ptr->nStates);
}

Communicator_internal Environment::create_communicator(
  const MPI_Comm workersComm,
  const int socket, const bool bSpawn)
{
  assert(socket>0);
  Communicator_internal comm(workersComm,socket,bSpawn,&settings.generators[0]);
  comm.set_exec_path(settings.launchfile);
  comm_ptr = &comm;

  if(settings.workers_rank>0) // aka not a master
  {
    #ifdef INTERNALAPP
      settings.nWorkers = settings.workers_size-1; //one is the master

      if(settings.nWorkers % settings.workersPerEnv != 0)
        die("Number of ranks does not match app\n");

      int workerGroup = (settings.workers_rank-1) / settings.workersPerEnv;

      MPI_Comm app_com;
      MPI_Comm_split(workersComm, workerGroup, settings.workers_rank, &app_com);

      comm.set_params_file(settings.appSettings);
      comm.set_nstepp_file(settings.nStepPappSett);
      comm.set_folder_path(settings.setupFolder);
      comm.set_application_mpicom(app_com, workerGroup);
      comm.ext_app_run(); //worker rank will remain here for ever
    #else
      //worker will fork and create environment
      comm_ptr->launch();
      //parent will stay here and set up communication between sim and master
      setDims();
    #endif
  }
  else  // master
  {
    setDims();
  }
  comm.update_state_action_dims(sI.dim, aI.dim);
  return comm;
}

Environment::~Environment() {
  for (auto & trash : agents)
    _dispose_object(trash);
}

bool Environment::predefinedNetwork(Builder & input_net) const
{
  // this function is to be filled by child classes
  // to implement convolutional models
  return false;
}

void Environment::commonSetup()
{
  assert(settings.nWorkers > 0);
  assert(nAgentsPerRank > 0);
  nAgents = nAgentsPerRank * settings.nWorkers;
  settings.nAgents = nAgents;

  if(sI.dim == 0) sI.dim = sI.inUse.size();
  if(sI.inUse.size() == 0) {
    if(settings.world_rank == 0)
    printf("Unspecified whether state vector components are available to learner, assumed yes\n");
    sI.inUse = vector<bool>(sI.dim, true);
  }
  if(sI.dim not_eq sI.inUse.size()) { die("must be equal"); }
  if(sI.dim == 0) {
    die("State vector dimensionality cannot be zero at this point");
  }

  sI.dimUsed = 0;
  for (Uint i=0; i<sI.inUse.size(); i++) if (sI.inUse[i]) sI.dimUsed++;

  if(settings.world_rank == 0)
  printf("State has %d component, %d in use\n", sI.dim, sI.dimUsed);

  aI.updateShifts();

  if(0 == aI.bounded.size()) {
    aI.bounded = vector<bool>(aI.dim, false);
    if(settings.world_rank == 0)
    printf("Unspecified whether action space is bounded: assumed not\n");
  } else assert(aI.bounded.size() == aI.dim);

  agents.resize(std::max(nAgents, (Uint) 1), nullptr);
  for(Uint i=0; i<nAgents; i++) agents[i] = new Agent(i, sI, aI);

  assert(sI.scale.size() == sI.mean.size());
  assert(sI.mean.size()==0 || sI.mean.size()==sI.dim);
  for (Uint i=0; i<sI.scale.size(); i++) assert(positive(sI.scale[i]));
}

bool Environment::pickReward(const Agent& agent)
{
  return agent.Status == 2;
}

Uint Environment::getNdumpPoints()
{
  return 0;
}

Rvec Environment::getDumpState(Uint k)
{
  return Rvec();
}
