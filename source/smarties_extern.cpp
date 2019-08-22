//==============================================================================
//
// smarties_extern.cpp
//
// This file defines the C/C++ functions that interface Smarties with Fortran
// code.  The respective Fortran subroutines are located in
// 'include/smarties.f90'.
//
// ****************************************************************************
// ********** There should be no need to modify any of the following **********
// **** If any of the following is edited, the corresponding F90 function *****
// ****** located in 'include/smarties.f90' should be edited accordingly ******
// ****************************************************************************
//
// Copyright (c) 2019 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
// Distributed under the terms of the MIT license.
//
//=============================================================================


//=============================================================================
#include "../include/smarties_extern.h"

//=============================================================================
extern "C" void smarties_sendInitState(void*const ptr2comm,
  const double*const S, const int state_dim, const int agentID)
{
  const std::vector<double> svec(S, S + state_dim);
  static_cast<smarties::Communicator*>(ptr2comm)->sendInitState(svec, agentID);
}

extern "C" void smarties_sendTermState(void*const ptr2comm,
  const double*const S, const int state_dim, const double R, const int agentID)
{
  const std::vector<double> svec(S, S + state_dim);
  static_cast<smarties::Communicator*>(ptr2comm)->sendTermState(svec, R, agentID);
}

extern "C" void smarties_sendLastState(void*const ptr2comm,
  const double*const S, const int state_dim, const double R, const int agentID)
{
  const std::vector<double> svec(S, S + state_dim);
  static_cast<smarties::Communicator*>(ptr2comm)->sendLastState(svec, R, agentID);
}

extern "C" void smarties_sendState(void*const ptr2comm,
  const double*const S, const int state_dim, const double R, const int agentID)
{
  const std::vector<double> svec(S, S + state_dim);
  static_cast<smarties::Communicator*>(ptr2comm)->sendState(svec, R, agentID);
}

extern "C" void smarties_recvAction(void*const ptr2comm,
  double*const A, const int action_dim, const int agentID)
{
  const std::vector<double> avec =
    static_cast<smarties::Communicator*>(ptr2comm)->recvAction(agentID);
  assert(action_dim == static_cast<int>(avec.size()));
  std::copy(avec.begin(), avec.end(), A);
}
//=============================================================================

//=============================================================================
extern "C" void smarties_set_num_agents(void*const ptr2comm,
  const int num_agents)
{
  static_cast<smarties::Communicator*>(ptr2comm)->set_num_agents(num_agents);
}

extern "C" void smarties_set_state_action_dims(void*const ptr2comm,
  const int state_dim, const int action_dim, const int agent_id)
{
  static_cast<smarties::Communicator*>(ptr2comm)->set_state_action_dims(
    state_dim, action_dim, agent_id);
}

extern "C" void smarties_set_action_scales_default(void*const ptr2comm,
  const double* const upper_scale, const double* const lower_scale,
  const int are_bounds, const int action_dim, const int agent_id)
{
  const std::vector<double> upper(upper_scale, upper_scale + action_dim);
  const std::vector<double> lower(lower_scale, lower_scale + action_dim);
  static_cast<smarties::Communicator*>(ptr2comm)->set_action_scales(
    upper, lower, are_bounds, agent_id);
}

extern "C" void smarties_set_action_scales_pointer(void*const ptr2comm,
  const double* const upper_scale, const double* const lower_scale,
  const int*const are_bounds, const int action_dim, const int agent_id)
{
  const std::vector<double> upper(upper_scale, upper_scale + action_dim);
  const std::vector<double> lower(lower_scale, lower_scale + action_dim);
  const std::vector<bool>  bounds( are_bounds,  are_bounds + action_dim);
  static_cast<smarties::Communicator*>(ptr2comm)->set_action_scales(
    upper, lower, bounds, agent_id);
}

extern "C" void smarties_set_action_options_default(void*const ptr2comm,
  const int noptions, const int agent_id)
{
  static_cast<smarties::Communicator*>(ptr2comm)->set_action_options(
    noptions, agent_id);
}

extern "C" void smarties_set_action_options_dim(void*const ptr2comm,
  const int*const noptions, const int action_dim, const int agent_id)
{
  const std::vector<int> optionsvec(noptions, noptions + action_dim);
  static_cast<smarties::Communicator*>(ptr2comm)->set_action_options(
    optionsvec, agent_id);
}

extern "C" void smarties_set_state_observable(void*const ptr2comm,
  const int*const bobservable, const int state_dim, const int agent_id)
{
  const std::vector<bool> optionsvec(bobservable, bobservable + state_dim);
  static_cast<smarties::Communicator*>(ptr2comm)->set_state_observable(
    optionsvec, agent_id);
}

extern "C" void smarties_set_state_scales(void*const ptr2comm,
  const double*const upper_scale, const double*const lower_scale,
  const int state_dim, const int agent_id)
{
  const std::vector<double> upper(upper_scale, upper_scale + state_dim);
  const std::vector<double> lower(lower_scale, lower_scale + state_dim);
  static_cast<smarties::Communicator*>(ptr2comm)->set_state_scales(
    upper, lower, agent_id);
}

extern "C" void smarties_set_is_partially_observable(void*const ptr2comm,
  const int agent_id)
{
  static_cast<smarties::Communicator*>(ptr2comm)->set_is_partially_observable(agent_id);
}

extern "C" void smarties_finalize_problem_description(void*const ptr2comm)
{
  static_cast<smarties::Communicator*>(ptr2comm)->finalize_problem_description();
}

extern "C" void smarties_env_has_distributed_agents(void*const ptr2comm)
{
  static_cast<smarties::Communicator*>(ptr2comm)->env_has_distributed_agents();
}

extern "C" void smarties_agents_define_different_MDP(void*const ptr2comm)
{
  static_cast<smarties::Communicator*>(ptr2comm)->agents_define_different_MDP();
}

extern "C" void smarties_disableDataTrackingForAgents(void*const ptr2comm,
  const int agentStart, const int agentEnd)
{
  static_cast<smarties::Communicator*>(ptr2comm)->disableDataTrackingForAgents(
    agentStart, agentEnd);
}

extern "C" void smarties_set_preprocessing_conv2d(void*const ptr2comm,
  const int input_width, const int input_height, const int input_features,
  const int kernels_num, const int filters_size, const int stride,
  const int agentID)
{
  static_cast<smarties::Communicator*>(ptr2comm)->set_preprocessing_conv2d(input_width,
    input_height, input_features, kernels_num, filters_size, stride, agentID);
}

extern "C" void smarties_set_num_appended_past_observations(void*const ptr2comm,
  const int n_appended, const int agentID)
{
  static_cast<smarties::Communicator*>(ptr2comm)->set_num_appended_past_observations(
    n_appended, agentID);
}

//=============================================================================
