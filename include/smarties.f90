!
!  cart_pole.f90
!  cart-pole
!
!  Created by Jacopo Canton on 31/01/19
!  Copyright (c) 2019 Jacopo Canton. All rights reserved.
!
!  This file contains the cart pole test case for Smarties in Fortran.
!   - The first module, `class_cartPole`, defines the environment and the
!   simulation parameters.
!   - The second module, `fortran_smarties`, contains the interface to the C++
!   code as well as the main function `fortran_app_main` called by Smarties.
!
!==============================================================================


!==============================================================================

module smarties
  ! Module interfacing with Smarties

  implicit none

  include 'mpif.h'

  interface

    subroutine smarties_sendInitState(
        ptr2comm, state, state_dim, agentID) &
        bind(c, name='smarties_sendInitState')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: state
      integer(c_int), intent(in), value :: state_dim
      integer(c_int), intent(in), value :: agentID
    end subroutine smarties_sendInitState

    subroutine smarties_sendState(
        ptr2comm, state, state_dim, reward, agentID) &
        bind(c, name='smarties_sendState')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: state
      integer(c_int), intent(in), value :: state_dim
      real(c_double), intent(in), value :: reward
      integer(c_int), intent(in), value :: agentID
    end subroutine smarties_sendState

    subroutine smarties_sendTermState(
        ptr2comm, state, state_dim, reward, agentID) &
        bind(c, name='smarties_sendTermState')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: state
      integer(c_int), intent(in), value :: state_dim
      real(c_double), intent(in), value :: reward
      integer(c_int), intent(in), value :: agentID
    end subroutine smarties_sendTermState

    subroutine smarties_sendLastState(
        ptr2comm, state, state_dim, reward, agentID) &
        bind(c, name='smarties_sendLastState')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: state
      integer(c_int), intent(in), value :: state_dim
      real(c_double), intent(in), value :: reward
      integer(c_int), intent(in), value :: agentID
    end subroutine smarties_sendLastState

    subroutine smarties_recvAction(
        ptr2comm, action, action_dim, agentID) &
        bind(c, name='smarties_recvAction')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: action
      integer(c_int), intent(in), value :: action_dim
      integer(c_int), intent(in), value :: agentID
    end subroutine smarties_recvAction

    subroutine smarties_set_num_agents(ptr2comm, num_agents) &
        bind(c, name='smarties_set_num_agents')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      integer(c_int), intent(in), value :: num_agents
    end subroutine smarties_set_num_agents

    subroutine smarties_set_state_action_dims(
        ptr2comm, state_dim, action_dim, agentID) &
        bind(c, name='smarties_set_state_action_dims')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      integer(c_int), intent(in), value :: state_dim
      integer(c_int), intent(in), value :: action_dim
      integer(c_int), intent(in), value :: agentID
    end subroutine smarties_set_state_action_dims

    subroutine smarties_set_action_scales(ptr2comm,
        upper_act_bound, lower_act_bound, bounded, action_dim, agentID) &
        bind(c, name='smarties_set_action_scales')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),     intent(in), value :: ptr2comm
      type(c_ptr),     intent(in), value :: upper_act_bound
      type(c_ptr),     intent(in), value :: lower_act_bound
      logical(c_bool), intent(in), value :: bounded
      integer(c_int),  intent(in), value :: action_dim
      integer(c_int),  intent(in), value :: agentID
    end subroutine smarties_set_action_scales

    subroutine smarties_set_action_scales(ptr2comm,
        upper_act_bound, lower_act_bound, bounded, action_dim, agentID) &
        bind(c, name='smarties_set_action_scales')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),     intent(in), value :: ptr2comm
      type(c_ptr),     intent(in), value :: upper_act_bound
      type(c_ptr),     intent(in), value :: lower_act_bound
      type(c_ptr),     intent(in), value :: bounded
      integer(c_int),  intent(in), value :: action_dim
      integer(c_int),  intent(in), value :: agentID
    end subroutine smarties_set_action_scales

    subroutine smarties_set_action_options(ptr2comm, num_options, agentID) &
        bind(c, name='smarties_set_action_options')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),     intent(in), value :: ptr2comm
      integer(c_int),  intent(in), value :: num_options
      integer(c_int),  intent(in), value :: agentID
    end subroutine smarties_set_action_options

    subroutine smarties_set_action_options(
        ptr2comm, num_options, action_dim, agentID) &
        bind(c, name='smarties_set_action_options')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),     intent(in), value :: ptr2comm
      type(c_ptr),     intent(in), value :: num_options
      integer(c_int),  intent(in), value :: action_dim
      integer(c_int),  intent(in), value :: agentID
    end subroutine smarties_set_action_options

    subroutine smarties_set_state_observable(
        ptr2comm, b_observable, state_dim, agentID) &
        bind(c, name='smarties_set_state_observable')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: b_observable
      integer(c_int), intent(in), value :: state_dim
      real(c_int),    intent(in), value :: agentID
    end subroutine smarties_set_state_observable

    subroutine smarties_set_state_scales(
        ptr2comm, upper_state_bound, lower_state_bound, state_dim, agentID) &
        bind(c, name='smarties_set_state_scales')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      type(c_ptr),    intent(in), value :: upper_state_bound
      type(c_ptr),    intent(in), value :: lower_state_bound
      integer(c_int), intent(in), value :: state_dim
      real(c_int),    intent(in), value :: agentID
    end subroutine smarties_set_state_scales

    subroutine smarties_set_is_partially_observable(ptr2comm, agentID) &
        bind(c, name='smarties_set_is_partially_observable')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      real(c_int),    intent(in), value :: agentID
    end subroutine smarties_set_is_partially_observable

    subroutine smarties_finalize_problem_description(ptr2comm) &
        bind(c, name='smarties_finalize_problem_description')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
    end subroutine smarties_finalize_problem_description

    subroutine smarties_env_has_distributed_agents(ptr2comm) &
        bind(c, name='smarties_env_has_distributed_agents')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
    end subroutine smarties_env_has_distributed_agents

    subroutine smarties_agents_define_different_MDP(ptr2comm) &
        bind(c, name='smarties_agents_define_different_MDP')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
    end subroutine smarties_agents_define_different_MDP

    subroutine smarties_disableDataTrackingForAgents(
        ptr2comm, agentStart, agentEnd) &
        bind(c, name='smarties_disableDataTrackingForAgents')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      real(c_int),    intent(in), value :: agentStart
      real(c_int),    intent(in), value :: agentEnd
    end subroutine smarties_disableDataTrackingForAgents

    subroutine smarties_set_preprocessing_conv2d(ptr2comm,
        input_width, input_height, input_features,
        kernels_num, filters_size, stride, agentID) &
        bind(c, name='smarties_set_preprocessing_conv2d')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      real(c_int),    intent(in), value :: input_width
      real(c_int),    intent(in), value :: input_height
      real(c_int),    intent(in), value :: input_features
      real(c_int),    intent(in), value :: kernels_num
      real(c_int),    intent(in), value :: filters_size
      real(c_int),    intent(in), value :: stride
      real(c_int),    intent(in), value :: agentID
    end subroutine smarties_set_preprocessing_conv2d

    subroutine smarties_set_num_appended_past_observations(
        ptr2comm, n_appended, agentID) &
        bind(c, name='smarties_set_num_appended_past_observations')
      use, intrinsic :: iso_c_binding
      implicit none
      type(c_ptr),    intent(in), value :: ptr2comm
      real(c_int),    intent(in), value :: n_appended
      real(c_int),    intent(in), value :: agentID
    end subroutine smarties_set_num_appended_past_observations

  end interface

end module fortran_smarties
