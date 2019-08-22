#ifndef smarties_extern_h
#define smarties_extern_h

#include "../source/Communicator.h"

//==============================================================================
//
// smarties_extern.h
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

#define VISIBLE __attribute__((visibility("default")))

//=============================================================================
extern "C" VISIBLE void smarties_sendInitState(void*const ptr2comm,
  const double*const S, const int state_dim, const int agentID);

extern "C" VISIBLE void smarties_sendTermState(void*const ptr2comm,
  const double*const S, const int state_dim, const double R, const int agentID);

extern "C" VISIBLE void smarties_sendLastState(void*const ptr2comm,
  const double*const S, const int state_dim, const double R, const int agentID);

extern "C" VISIBLE void smarties_sendState(void*const ptr2comm,
  const double*const S, const int state_dim, const double R, const int agentID);

extern "C" VISIBLE void smarties_recvAction(void*const ptr2comm,
  double*const A, const int action_dim, const int agentID);
//=============================================================================

//=============================================================================
extern "C" VISIBLE void smarties_set_num_agents(void*const ptr2comm,
  const int num_agents);

extern "C" VISIBLE void smarties_set_state_action_dims(void*const ptr2comm,
  const int state_dim, const int action_dim, const int agent_id);

extern "C" VISIBLE void smarties_set_action_scales_default(void*const ptr2comm,
  const double* const upper_scale, const double* const lower_scale,
  const int are_bounds, const int action_dim, const int agent_id);

extern "C" VISIBLE void smarties_set_action_scales_pointer(void*const ptr2comm,
  const double* const upper_scale, const double* const lower_scale,
  const int*const are_bounds, const int action_dim, const int agent_id);

extern "C" VISIBLE void smarties_set_action_options_default(void*const ptr2comm,
  const int noptions, const int agent_id);

extern "C" VISIBLE void smarties_set_action_options_dim(void*const ptr2comm,
  const int*const noptions, const int action_dim, const int agent_id);

extern "C" VISIBLE void smarties_set_state_observable(void*const ptr2comm,
  const int*const bobservable, const int state_dim, const int agent_id);

extern "C" VISIBLE void smarties_set_state_scales(void*const ptr2comm,
  const double*const upper_scale, const double*const lower_scale,
  const int state_dim, const int agent_id);

extern "C" VISIBLE void smarties_set_is_partially_observable(
  void*const ptr2comm, const int agent_id);

extern "C" VISIBLE void smarties_finalize_problem_description(void*const ptr2comm);

extern "C" VISIBLE void smarties_env_has_distributed_agents(void*const ptr2comm);

extern "C" VISIBLE void smarties_agents_define_different_MDP(void*const ptr2comm);

extern "C" VISIBLE void smarties_disableDataTrackingForAgents(
  void*const ptr2comm, const int agentStart, const int agentEnd);

extern "C" VISIBLE void smarties_set_preprocessing_conv2d(void*const ptr2comm,
  const int input_width, const int input_height, const int input_features,
  const int kernels_num, const int filters_size, const int stride,
  const int agentID);

extern "C" VISIBLE void smarties_set_num_appended_past_observations(
  void*const ptr2comm, const int n_appended, const int agentID);

// TODO DISCUSS PRNG, RETURN VALUES
//=============================================================================

#undef VISIBLE
#endif
