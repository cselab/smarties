#include "Communicator.h"
#include <vector>
#include "mpi.h"

//=============================================================================
// Entry point into E.g. fortran code.
extern "C" void extern_app_main(const void* rlcomm, const int f_mpicomm);
//=============================================================================

//=============================================================================
// Program entry point
int app_main(
  Communicator*const rlcomm, // communicator with smarties
  MPI_Comm c_mpicomm,        // mpi_comm that mpi-based apps can use (C handle)
  int argc, char**argv,      // arguments read from app's runtime settings file
)
{
  std::cout << "C++ side begins" << std::endl;

  // Convert the C handle to the MPI communicator to a Fortran handle
  MPI_Fint f_mpicomm;
  f_mpicomm = MPI_Comm_c2f(c_mpicomm);

  fortran_app_main(rlcomm, f_mpicomm);

  std::cout << "C++ side ends" << std::endl;
  return 0;
} // main
//=============================================================================

//=============================================================================
extern "C" void smarties_sendInitState(void*const ptr2comm,
  const double*const S, const int state_dim, const int agentID
)
{
  const std::vector<double> svec(S, S + state_dim);
  static_cast<Communicator*>(ptr2comm)->sendInitState(svec, agentID);
}

extern "C" void smarties_sendTermState(void*const ptr2comm,
  const double*const S, const int state_dim, const double R, const int agentID
)
{
  const std::vector<double> svec(S, S + state_dim);
  static_cast<Communicator*>(ptr2comm)->sendTermState(svec, R, agentID);
}

extern "C" void smarties_sendLastState(void*const ptr2comm,
  const double*const S, const int state_dim, const double R, const int agentID
)
{
  const std::vector<double> svec(S, S + state_dim);
  static_cast<Communicator*>(ptr2comm)->sendLastState(svec, R, agentID);
}

extern "C" void smarties_sendState(void*const ptr2comm,
  const double*const S, const int state_dim, const double R, const int agentID
)
{
  const std::vector<double> svec(S, S + state_dim);
  static_cast<Communicator*>(ptr2comm)->sendState(svec, R, agentID);
}

extern "C" void smarties_recvAction(void*const ptr2comm,
  double*const A, const int action_dim, const int agentID
)
{
  const std::vector<double> avec =
    static_cast<Communicator*>(ptr2comm)->recvAction(agentID);
  assert(action_dim == static_cast<int>(avec));
  std::copy(avec.begin(), avec.end(), A);
}
//=============================================================================

//=============================================================================
extern "C" void smarties_set_num_agents(void*const ptr2comm,
  const int num_agents
)
{
  static_cast<Communicator*>(ptr2comm)->set_num_agents(num_agents);
}

extern "C" void smarties_set_state_action_dims(void*const ptr2comm,
  const int state_dim, const int action_dim, const int agent_id
)
{
  static_cast<Communicator*>(ptr2comm)->set_state_action_dims(
    state_dim, action_dim, agent_id);
}

extern "C" void smarties_set_action_scales(void*const ptr2comm,
  const double* const upper_scale, const double* const lower_scale,
  const int are_bounds, const int action_dim, const int agent_id
)
{
  const std::vector<double> upper(upper_scale, upper_scale + action_dim);
  const std::vector<double> lower(lower_scale, lower_scale + action_dim);
  static_cast<Communicator*>(ptr2comm)->set_action_scales(
    upper, lower, are_bounds, agent_id);
}

extern "C" void smarties_set_action_scales(void*const ptr2comm,
  const double* const upper_scale, const double* const lower_scale,
  const int*const are_bounds, const int action_dim, const int agent_id
)
{
  const std::vector<double> upper(upper_scale, upper_scale + action_dim);
  const std::vector<double> lower(lower_scale, lower_scale + action_dim);
  const std::vector<bool>  bounds( are_bounds,  are_bounds + action_dim);
  static_cast<Communicator*>(ptr2comm)->set_action_scales(
    upper, lower, bounds, agent_id);
}

extern "C" void smarties_set_action_options(void*const ptr2comm,
  const int noptions, const int agent_id
)
{
  static_cast<Communicator*>(ptr2comm)->set_action_options(
    noptions, agent_id);
}

extern "C" void smarties_set_action_options(void*const ptr2comm,
  const int*const noptions, const int action_dim, const int agent_id
)
{
  const std::vector<int> optionsvec(noptions, noptions + action_dim);
  static_cast<Communicator*>(ptr2comm)->set_action_options(
    optionsvec, agent_id);
}

extern "C" void smarties_set_state_observable(void*const ptr2comm,
  const int*const bobservable, const int state_dim, const int agent_id
)
{
  const std::vector<bool> optionsvec(bobservable, bobservable + state_dim);
  static_cast<Communicator*>(ptr2comm)->set_state_observable(
    bobservable, agent_id);
}

extern "C" void smarties_set_state_scales(void*const ptr2comm,
  const double*const upper_scale, const double*const lower_scale,
  const int state_dim, const int agent_id
)
{
  const std::vector<double> upper(upper_scale, upper_scale + state_dim);
  const std::vector<double> lower(lower_scale, lower_scale + state_dim);
  static_cast<Communicator*>(ptr2comm)->set_state_scales(
    upper, lower, agent_id);
}

extern "C" void smarties_set_is_partially_observable(void*const ptr2comm,
  const int agent_id
)
{
  static_cast<Communicator*>(ptr2comm)->set_is_partially_observable(agent_id);
}

extern "C" void smarties_finalize_problem_description(void*const ptr2comm)
{
  static_cast<Communicator*>(ptr2comm)->finalize_problem_description();
}

extern "C" void smarties_env_has_distributed_agents(void*const ptr2comm)
{
  static_cast<Communicator*>(ptr2comm)->env_has_distributed_agents();
}

extern "C" void smarties_agents_define_different_MDP(void*const ptr2comm)
{
  static_cast<Communicator*>(ptr2comm)->agents_define_different_MDP();
}

extern "C" void smarties_disableDataTrackingForAgents(void*const ptr2comm,
  const int agentStart, const int agentEnd)
{
  static_cast<Communicator*>(ptr2comm)->disableDataTrackingForAgents(
    agentStart, agentEnd);
}

extern "C" void smarties_set_preprocessing_conv2d(void*const ptr2comm,
  const int input_width, const int input_height, const int input_features,
  const int kernels_num, const int filters_size, const int stride,
  const int agentID
)
{
  static_cast<Communicator*>(ptr2comm)->set_preprocessing_conv2d(input_width,
    input_height, input_features, kernels_num, filters_size, stride, agentID);
}

extern "C" void smarties_set_num_appended_past_observations(void*const ptr2comm,
  const int n_appended, const int agentID)
{
  static_cast<Communicator*>(ptr2comm)->set_num_appended_past_observations(
    n_appended, agentID);
}

// TODO DISCUSS PRNG, RETURN VALUES
//=============================================================================
