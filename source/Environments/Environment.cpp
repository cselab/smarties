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
  assert(comm_ptr->getDimS()>=0);
  assert(comm_ptr->getDimA()>0);
  assert(comm_ptr->getNagents()>0);
  assert(comm_ptr->nDiscreteAct()>=0);

  sI.dim = comm_ptr->getDimS();
  aI.dim = comm_ptr->getDimA();
  nAgentsPerRank = comm_ptr->getNagents();
  aI.discrete = comm_ptr->nDiscreteAct();

  aI.values.resize(aI.dim);
  aI.bounded.resize(aI.dim, 0);
  sI.mean.resize(sI.dim);
  sI.scale.resize(sI.dim);
  sI.inUse.resize(sI.dim, 1);

  if(settings.world_rank==0) printf("State dimensionality : %d.",sI.dim);
  for (unsigned i=0; i<sI.dim; i++) {
    const bool inuse = comm_ptr->isStateObserved()[i] > 0.5;
    const double upper = comm_ptr->stateBounds()[i*2+0];
    const double lower = comm_ptr->stateBounds()[i*2+1];
    sI.inUse[i] = inuse;
    sI.mean[i]  = 0.5*(upper+lower);
    sI.scale[i] = 0.5*std::fabs(upper-lower);
    if(sI.scale[i]>=1e3 || sI.scale[i] < 1e-7) {
      if(settings.world_rank == 0) printf(" unbounded");
      sI.scale = Rvec(sI.dim, 1); sI.mean = Rvec(sI.dim, 0);
      break;
    }
    if(settings.world_rank==0 && not inuse)
      printf(" State component %u is hidden from the learner.", i);
  }
  if(settings.world_rank==0) printf("\nAction dimensionality : %d.",aI.dim);

  int k = 0;
  for (Uint i=0; i<aI.dim; i++) {
    aI.bounded[i]   = comm_ptr->actionOption()[i*2 +1] > 0.5;
    const int nvals = comm_ptr->actionOption()[i*2 +0];
    // if act space is continuous, only receive high and low val for each action
    assert(aI.discrete || nvals == 2);
    assert(nvals > 1);
    aI.values[i].resize(nvals);
    for(int j=0; j<nvals; j++)
      aI.values[i][j] = comm_ptr->actionBounds()[k++];

    const Real amax = aI.getActMaxVal(i), amin = aI.getActMinVal(i);
    if(settings.world_rank==0)
    printf(" [%u: %.1f:%.1f%s]", i, amin, amax, aI.bounded[i]?" (bounded)":"");
  }
  if(settings.world_rank==0) printf("\n");

  commonSetup(); //required
  assert(sI.dim == (Uint) comm_ptr->getDimS());
}

Communicator_internal Environment::create_communicator()
{
  Communicator_internal comm(settings);
  comm_ptr = &comm;

  #ifdef INTERNALAPP
  if(settings.workers_rank>0) // aka not a master
  {
    if( (settings.workers_size-1) % settings.workersPerEnv != 0)
      die("Number of ranks does not match app\n");

    int workerGroup = (settings.workers_rank-1) / settings.workersPerEnv;

    MPI_Comm app_com;
    MPI_Comm_split(settings.workersComm, workerGroup, settings.workers_rank,
      &app_com);
    comm.set_application_mpicom(app_com, workerGroup);
    comm.ext_app_run(); //worker rank will remain here for ever
  }
  if(settings.bSpawnApp)
   die("Learn rank cannot spawn an internally linked app. Use multiple ranks");
  #endif

  if(settings.bSpawnApp) { comm_ptr->launch(); }
  setDims();

  comm.update_state_action_dims(sI.dim, aI.dim);
  return comm;
}

Environment::~Environment() {
  for (auto & trash : agents) _dispose_object(trash);
}

bool Environment::predefinedNetwork(Builder & input_net) const
{
  // this function is to be filled by child classes
  // to implement convolutional models
  return false;
}

void Environment::commonSetup()
{
  assert(nAgentsPerRank > 0);
  nAgents = nAgentsPerRank * settings.nWorkers_own;
  settings.nAgents = nAgents;

  if(sI.dim == 0) sI.dim = sI.inUse.size();
  if(sI.inUse.size() == 0 and sI.dim > 0) {
    if(settings.world_rank == 0)
    printf("Unspecified whether state vector components are available to learner, assumed yes\n");
    sI.inUse = std::vector<bool>(sI.dim, true);
  }
  if(sI.dim not_eq sI.inUse.size()) { die("must be equal"); }

  sI.dimUsed = 0;
  for (Uint i=0; i<sI.inUse.size(); i++) if (sI.inUse[i]) sI.dimUsed++;

  if(settings.world_rank == 0)
  printf("State has %d component, %d in use\n", sI.dim, sI.dimUsed);

  aI.updateShifts();

  if(0 == aI.bounded.size()) {
    aI.bounded = std::vector<bool>(aI.dim, false);
    if(settings.world_rank == 0)
      printf("Unspecified whether action space is bounded: assumed not\n");
  } else assert(aI.bounded.size() == aI.dim);

  agents.resize(nAgents, nullptr);
  for(Uint i=0; i<nAgents; i++) {
    const Uint workerID = i / nAgentsPerRank;
    const Uint localID = i % nAgentsPerRank;
    agents[i] = new Agent(i, sI, aI, workerID, localID);
  }

  assert(sI.scale.size() == sI.mean.size());
  assert(sI.mean.size()==0 || sI.mean.size()==sI.dim);
  for (Uint i=0; i<sI.scale.size(); i++) assert(positive(sI.scale[i]));
}

bool Environment::pickReward(const Agent& agent) const
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

Uint Environment::getNumberRewardParameters() {
  return 0;
}

// compute the reward given a certain state and param vector
Real Environment::getReward(const std::vector<memReal> s, const Rvec params)
{
  return 0;
}

// compute the gradient of the reward
Rvec Environment::getRewardGrad(const std::vector<memReal> s, const Rvec params)
{
  return Rvec();
}
